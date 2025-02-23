#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module simulates a chase between a Robber Turtle and a Police Turtle,
similar to Goal 4.

The Police Turtle's velocity as capped at half the average velocity of the
Robber Turtle.

Therefore, a planning element is required.
Assumption: The Police Turtle has access to the Robber Turtle's trajectory
in (x, y, time) and can therefore determine the point on said trajectory
that it can capture the Robber Turtle earliest.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen
from random import uniform
from math import pi
from flyt_task import utils


class RobberTurtle:
    """
    A controller for TurtleSim turtles using PD - control with velocity profiling.

    This class implements a control system that guides a turtle to a goal position
    using a PD - Controller. The resulting velocity vector is decomposed
    into the turtle's local frame for forward and strafe control.

    A trajectory is generated as a list of timed waypoints (x, y, t), where
    the Turtle is expected to reach waypoint (x, y) at time t relative to the
    time it starts the circular trajectory.

    Ensures that the circular trajectory of a given radius is completed
    in a given time.

    This class also uses throttling to publish the actual and noisy Pose data
    of the Turtle every 5 seconds.
    """

    def __init__(self, radius=3.5, time=15.0):
        """
        Args:
            radius (float): Radius of the circular trajectory (TurtleSim coordinate units)
            time (float): Time taken by the turtle to complete a circle (seconds)
        """
        rospy.init_node("turtle", anonymous=True)
        self.radius = radius
        self.time = time

        """Use TurtleSim Kill and Spawn services to start at a point on a circle of given radius."""
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.kill("turtle1")
        
        # Spawn at (center_x + radius, center_y)
        spawn_x, spawn_y = 5.5 + self.radius, 5.5
        self.spawn(spawn_x, spawn_y, 0, "turtle2")

        """Use TurtleSim SetPen service to show the Robber Turtle's trajectory in red."""
        rospy.wait_for_service('turtle2/set_pen')
        self.set_pen = rospy.ServiceProxy('turtle2/set_pen', SetPen)
        self.set_pen(255, 0, 0, 3, 0)

        # PD - Control parameters
        self.Kp = 15.0
        self.Kd = 3.5

        # Error Term
        self.prev_distance_error = 0.0

        # Trajectory of the circular path (x, y, time)
        self.trajectory = utils.generate_circular_trajectory(self.time, self.radius)

        # Track start time of drawing
        self.start_time = None

        """The turtle spawns at 0th waypoint at 0 time"""

        # Track next waypoint
        self.current_waypoint = 1
        
        # Goal pose to be initialized as first waypoint
        self.goal = Pose()
        self.goal.x = self.trajectory[1][0]
        self.goal.y = self.trajectory[1][1]

        # Current pose data
        self.current_pose = Pose()
        self.is_caught = False

        # Publisher for command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)

        # Subscriber for pose
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup a publisher for real pose (throttled)
        self.throttled_pub = rospy.Publisher('rt_real_pose', Pose, queue_size=10)
        self.last_published_time = rospy.Time.now()

        # Setup a publisher for noisy pose (throttled)
        self.noisy_pub = rospy.Publisher('rt_noisy_pose', Pose, queue_size=10)
        self.noisy_pose = Pose()

    def pose_callback(self, data):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose
        until RT is caught.

        Publishes the real Pose and the noisy Pose every 5 seconds.

        Args:
            data (Pose): incoming pose data from TurtleSim
        """
        self.current_pose = data

        # Set start time when the first pose callback occurs
        if self.start_time is None:
            self.start_time = rospy.Time.now()

        if not self.is_caught:
            self.draw_circle()

        # Publish the current pose of the turtle every 5 seconds
        if (rospy.Time.now() - self.last_published_time).to_sec() >= 5.0:
            self.throttled_pub.publish(self.current_pose)

            # Add random Gaussian noise (std dev = 10) to the current position
            self.noisy_pose = utils.add_random_noise(self.current_pose)
            self.noisy_pub.publish(self.noisy_pose)

            self.last_published_time = rospy.Time.now()
    
    def draw_circle(self):
        """
        Uses the next waypoint as a goal and implements a similar PD - Controller as before to move to it.
        Once the waypoint is reached, it updates the goal to the next waypoint.

        If the turtle reaches a waypoint earlier than expected, it will wait till it is actually expected to be there.
        """
        # Calculate error
        distance_error = utils.calculate_distance(self.goal, self.current_pose)
        angle_error = utils.calculate_angle(self.goal, self.current_pose)
        
        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # PD - Control for translation
        target_velocity = self.Kp * distance_error + self.Kd * distance_error_derivative
        
        velocity_magnitude = target_velocity
        velocity_direction = angle_error

        # Create a Twist message
        cmd = Twist()

        # Check if the current waypoint has been reached
        if distance_error < 0.1:
            # Calculate time expected as well as time actually taken to reach this waypoint
            time_elapsed = (rospy.Time.now() - self.start_time).to_sec()
            time_expected = self.trajectory[self.current_waypoint][2]

            # Reset start time upon completion of the circle
            if self.current_waypoint == 0:
                rospy.loginfo(f"Circle completed in {time_elapsed:.2f} seconds!")
                self.start_time = rospy.Time.now()

            else:
                while (rospy.Time.now() - self.start_time).to_sec() <= time_expected:
                    # If the turtle is early to this waypoint, hold it still till the time it is actually expected to be there
                    self.cmd_pub.publish(cmd) # Hold turtle still

            """Update waypoint and continue circular trajectory infinitely."""
            self.current_waypoint = (self.current_waypoint + 1) % 360
            self.goal.x = self.trajectory[self.current_waypoint][0]
            self.goal.y = self.trajectory[self.current_waypoint][1]

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals
        self.cmd_pub.publish(cmd)

    def stop(self):
        """Stops the Robber Turtle by setting is_caught flag and publishing zero velocity."""
        self.is_caught = True
        cmd = Twist()
        self.cmd_pub.publish(cmd)


class PoliceTurtle:
    """
    Controller for the pursuing Police Turtle that chases the Robber Turtle.

    This class implements a P - controlled pursuit behavior with a velocity limit.
    The Police Turtle spawns at a random point in the TurtleSim frame and has access
    to the Robber Turtle's trajectory.

    The velocity of the Police Turtle is capped at half the average veloity of the
    Robber Turtle, which completes a circle of a specified radius in specified time.

    This class uses the trajectory information and velocity constraint to
    estimate the optimal point of intercept on the Robber Turtle's trajectory
    and uses this point as a goal instead of the Robber Turtle's last known pose.
    """
    def __init__(self, robber_turtle: RobberTurtle):
        """
        Args:
            robber_turtle (RobberTurtle): Reference to the Robber Turtle instance
        """
        self.robber = robber_turtle

        """Use the TurtleSim Spawn service to spawn Police turtle at a random position."""
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        spawn_x = uniform(0.5, 10.5)
        spawn_y = uniform(0.5, 10.5)
        self.spawn(spawn_x, spawn_y, 0, "turtle3")

        rospy.loginfo(f"Police Turtle spawned at x: {spawn_x:.3f}, y: {spawn_y:.3f}")

        # P - Control parameters
        self.Kp = 10.0

        # Current pose of the Police Turtle
        self.current_pose = Pose()

        # Limit the velocity of the Police Turtle (1/2 * average RT velocity)
        self.max_velocity = (2 * pi * self.robber.radius / self.robber.time) / 2

        # Point at which intercept is attempted
        self.goal = None

        # Track the last known pose of the Robber Turtle
        self.robber_pose = None

        # Publisher for command velocity
        self.cmd_pub = rospy.Publisher('turtle3/cmd_vel', Twist, queue_size=10)

        # Subscriber for pose
        self.pose_sub = rospy.Subscriber('turtle3/pose', Pose, self.pose_callback)

        # Setup subscriber to get the Pose of the Robber Turtle every 5 seconds
        self.robber_sub = rospy.Subscriber('rt_real_pose', Pose, self.robber_callback)

    def robber_callback(self, data: Pose):
        """
        Updates the last known position of the Robber Turtle (every 5 seconds).

        Args:
            data (Pose): incoming real pose of the Robber Turtle
        """
        self.robber_pose = data
        rospy.loginfo("Robber Turtle pose recieved...")

    def pose_callback(self, data: Pose):
        """
        Updates current pose of the Police Turtle and checks if the Robber Turtle is caught.

        Args:
            data (Pose): incoming real pose of the Robber Turtle
        """
        self.current_pose = data

        """
        The chase starts only when the Police Turtle is informed
        of the Robber Turtle's pose and ends when the Robber Turtle
        is caught.
        """

        if not self.robber.is_caught and self.robber_pose is not None:
            self.chase_robber()

            # Calculate Euclidian distance between Police Turtle and actual Pose of the robber turtle
            distance = utils.calculate_distance(self.current_pose, self.robber.current_pose)
            
            """Stop both turtles when the Robber Turtle is caught (tolerance = radius * 0.1)"""

            if distance <= 0.4:
                self.stop()
                self.robber.stop()
                rospy.loginfo("Robber Turtle caught!")

    def evaluate_intercept(self, target_point: Pose, target_time: float) -> rospy.Time:
        """
        Evaluates if an intercept is possible at a given point.
        Returns estimated time of intercept if possible.
        
        Args:
            target_point (Pose): Point at which intercept is to be evaluated
            target_time (float): Time (relative from the start of the circular trajectory)
                                    taken by Robber Turtle to reach this point (seconds)

        Returns:
            rospy.Time or None: Estimated intercept time if intercept is possible, None if impossible
        """
        current_time = rospy.Time.now()
        
        distance_to_target = utils.calculate_distance(target_point, self.current_pose)
        time_needed = distance_to_target / self.max_velocity

        pt_arrival = current_time + rospy.Duration(time_needed)
        rt_arrival = self.robber.start_time + rospy.Duration(target_time)

        # Do not return points at which Police Turtle has to wait for Robber Turtle
        if abs((pt_arrival - rt_arrival).to_sec()) <= 0.025:
            return max(pt_arrival, rt_arrival)
        
        # If the intercept is not valid
        return None
    
    def find_optimal_intercept(self):
        """
        Searches through the Robber Turtle's trajectory to find point in the
        future with earliest estimated time of intercept (in a 30 second window).
        """
        current_time = rospy.Time.now()
        best_intercept = None
        best_intercept_time = current_time + rospy.Duration(15) # Catch within 15 seconds

        for x, y, t, in self.robber.trajectory:
            # Evaluate points only in the future
            if current_time < (self.robber.start_time + rospy.Duration(t)):
                target_point = Pose()
                target_point.x, target_point.y = x, y

                intercept_time = self.evaluate_intercept(target_point, t)
                if intercept_time is not None and intercept_time < best_intercept_time:
                    best_intercept = target_point
                    best_intercept_time = intercept_time

        return best_intercept
    
    def chase_robber(self):
        """
        Execute the pursuit control loop with velocity profiling and speed limitations.

        This method:
        1. Calculates control errors based on the planned intercept point
        2. Applies P - Control with an upper limit on velocity
        3. Transforms and publishes velocity commands
        4. Recalculates intercept point if Robber Turtle is not caught at current
            intercept point.
        """
        if self.goal is None:
            self.goal = self.find_optimal_intercept()

            if self.goal:
                rospy.loginfo(f"Optimal interept found at x={self.goal.x:.2f}, y={self.goal.y:.2f}")

            elif self.robber_pose is not None:
                self.goal = self.robber_pose
                rospy.logwarn("Couldn't finnd optimal intercept point, going to Robber's last known Pose!")
            
            else:
                rospy.loginfo("No optimal intercept or Robber Pose info available!")
                return

        # Calculate error
        distance_error = utils.calculate_distance(self.goal, self.current_pose)
        angle_error = utils.calculate_angle(self.goal, self.current_pose)

        # P - Control for targeted magintude of velocity
        target_velocity = self.Kp * distance_error

        velocity_magnitude = min(target_velocity, self.max_velocity)
        velocity_direction = angle_error

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals
        self.cmd_pub.publish(cmd)

        if distance_error < 0.1:
            rospy.logwarn("Couldn't catch turtle, re-calculating optimal intercept point...")
            self.goal = self.find_optimal_intercept()

            if self.goal:
                rospy.loginfo(f"New optimal intercept found at x={self.goal.x:.3f}, y={self.goal.y:.3f}")
            else:
                self.goal = self.robber_pose
                rospy.logwarn("Couldn't finnd optimal intercept point, going to Robber's last known Pose!")

    def stop(self):
        """
        Stops the Police Turtle by publishing zero velocity
        Sets the is_caught flag of the Robber Turtle.
        """
        self.robber.is_caught = True
        cmd = Twist()
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        robber_turtle = RobberTurtle(4.0, 15.0)
        rospy.sleep(10)
        police_turtle = PoliceTurtle(robber_turtle)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass