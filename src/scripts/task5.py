#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Similar setup as Goal 4.
The speed of the Police Turtle will be limited to 1/2 the speed of the Robber Turtle.
Therefore, a planning element is required for PT to be able to catch RT.
Assumption: PT has access to the trajectory of RT in (x, y, time).
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
    TurtleSim turtle turtle that implements a PD - Controller on the turtle's forward and strafe velocities.
    Generate waypoints for a circular trajectory (at every 1 degree) with variable radius and speed.

    Publishes the real Pose of the turtle as well as the real Pose with a random Gaussian noise, every 5 seconds.
    """
    def __init__(self, radius=3.5, time=15.0):
        """
        Args:
            radius (float): Radius of the circular trajectory
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

        self.center_x, self.center_y = 5.5, 5.5 # Center of TurtleSim frame
        
        # Spawn at (center_x + radius, center_y)
        spawn_x, spawn_y = self.center_x + self.radius, self.center_y
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

        """The turtle spawns at 0th waypoint at 0 time"""

        # Track start time of drawing
        self.start_time = None

        # Track next waypoint
        self.current_waypoint = 1
        
        # Goal pose to be initialized as first waypoint
        self.goal = Pose()
        self.goal.x = self.trajectory[1][0]
        self.goal.y = self.trajectory[1][1]

        # Current pose data
        self.current_pose = Pose()
        self.is_caught = False

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup publishers for throttled pose
        self.throttled_pub = rospy.Publisher('rt_real_pose', Pose, queue_size=10)
        self.last_published_time = rospy.Time.now()

        # Setup a publisher to publish noisy pose
        self.noisy_pub = rospy.Publisher('rt_noisy_pose', Pose, queue_size=10)
        self.noisy_pose = Pose()

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.

        Publishes the real Pose and the noisy Pose every 5 seconds.
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
        Uses the next waypoint as a goal and implements a PD - Controller to move to it.
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
                self.start_time = rospy.Time.now()
            else:
                while (rospy.Time.now() - self.start_time).to_sec() <= time_expected:
                    # If the turtle is early to this waypoint, hold it still till the time it is actually expected to be there
                    self.cmd_pub.publish(cmd) # Hold turtle still

            # Update waypoint
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
    TurtleSim turtle that implements a PD - Controller with an external
    acceleration / deceleration profile, to chase the Robber Turtle.
    
    Initialized with a reference to the Robber Turtle. This reference is used to limit the Police Turtle's velocity
    and allow it to acess the Robber Turtle's trajectory.

    Spawns at a random position and receives the Robber Turtle's pose every 5 seconds.
    """
    def __init__(self, robber_turtle):
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

        # PD - Control parameters
        self.Kp = 50.0
        self.Kd = 5.0

        # Error term
        self.prev_distance_error = None

        """
        Initializing the tracking variable for the previous distance error to zero causes the
        first published velocity to be unnaturally large.
        """

        # Current pose of the Police Turtle
        self.current_pose = Pose()

        # Limit the velocity of the Police Turtle
        self.max_velocity = 2 * pi * self.robber.radius / self.robber.time

        # Point at which intercept is attempted
        self.goal = None
        while self.goal is None:
            rospy.logwarn("Couldn't find intercept point! Checking again...")
            self.goal = self.find_optimal_intercept()
        rospy.loginfo(f"Optimal intercept found at x={self.goal.x:.3f}, y={self.goal.y:.3f}")

        # Track the last known pose of the Robber Turtle
        self.robber_pose = None

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle3/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle3/pose', Pose, self.pose_callback)

        # Setup subscriber to get the Pose of the Robber Turtle every 5 seconds
        self.robber_sub = rospy.Subscriber('rt_real_pose', Pose, self.robber_callback)

        # Define acceleration and deceleration limits
        self.acceleration_limit = 5.0
        self.deceleration_limit = 13.5

        # Scale limits to compensate for negligible dt values
        self.acceleration_limit *= 1000
        self.deceleration_limit *= 1000

        # Track current linear velocity
        self.current_linear_velocity = 0.0
        
        # Track the current time
        self.last_time = rospy.Time.now()

    def robber_callback(self, data):
        """Updates the last known position of the Robber Turtle (every 5 seconds)."""
        self.robber_pose = data
        rospy.loginfo("Robber Turtle pose recieved...")

    def pose_callback(self, data):
        """Updates current pose of the Police Turtle and checks if the Robber Turtle is caught."""
        self.current_pose = data
        self.current_linear_velocity = self.current_pose.linear_velocity

        if not self.robber.is_caught and self.robber_pose is not None:
            self.chase_robber()

            # Calculate Euclidian distance between Police Turtle and actual Pose of the robber turtle
            distance = utils.calculate_distance(self.current_pose, self.robber.current_pose)
            
            """Stop both turtles when the Robber Turtle is caught (tolerance = radius * 0.1)"""
            if distance <= 0.4:
                self.stop()
                self.robber.stop()
                rospy.loginfo("Robber Turtle caught!")

    def estimate_intercept(self, target_point, target_time):
        current_time = rospy.Time.now()

        if self.max_velocity <= 0:
            return None
        
        distance_to_target = utils.calculate_distance(target_point, self.current_pose)
        time_needed = distance_to_target / self.max_velocity

        pt_arrival = current_time + rospy.Duration(time_needed)
        rt_arrival = self.robber.start_time + rospy.Duration(target_time)

        if abs((pt_arrival - rt_arrival).to_sec()) <= 0.01:
            return max(pt_arrival, rt_arrival)
        
        return None
    
    def find_optimal_intercept(self):
        current_time = rospy.Time.now()
        best_intercept = None
        best_intercept_time = current_time + rospy.Duration(15) # Catch within 15 seconds

        for x, y, t, in self.robber.trajectory:
            if current_time < (self.robber.start_time + rospy.Duration(t)):
                target_point = Pose()
                target_point.x, target_point.y = x, y

                intercept_time = self.estimate_intercept(target_point, t)
                if intercept_time is not None and intercept_time < best_intercept_time:
                    best_intercept = target_point
                    best_intercept_time = intercept_time

        return best_intercept
    
    def chase_robber(self):
        """
        Uses PD - control to chase the Robber Turtle based on its last known position.
        Applies acceleration and deceleration limits.
        """
        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

        # Calculate error
        distance_error = utils.calculate_distance(self.goal, self.current_pose)
        angle_error = utils.calculate_angle(self.goal, self.current_pose)

        # Handle special case for first callback
        if self.prev_distance_error is None:
            self.prev_distance_error = distance_error
            self.current_linear_velocity = 0.0
            self.last_time = rospy.Time.now()
            return # Skip this control cycle

        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # PD - Control for targeted magintude of velocity
        target_velocity = (self.Kp * distance_error + self.Kd * distance_error_derivative)

        # Apply limits to change in velocity
        velocity_delta_required = target_velocity - self.current_linear_velocity
        if velocity_delta_required > 0: # acceleartion
            max_delta = self.acceleration_limit * dt
            velocity_delta_achieved = min(velocity_delta_required, max_delta)
        else: # deceleration
            max_delta = -self.deceleration_limit * dt
            velocity_delta_achieved = max(velocity_delta_required, max_delta)

        velocity_magnitude = self.current_linear_velocity + velocity_delta_achieved
        velocity_direction = angle_error

        velocity_magnitude = max(min(velocity_magnitude, self.max_velocity), 0.1)

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals
        self.cmd_pub.publish(cmd)

        if distance_error < 0.25:
            rospy.logwarn("Couldn't catch turtle, re-calculating optimal intercept point...")
            self.goal = None
            while self.goal is None:
                rospy.logwarn("Couldn't find intercept! Checking again...")
                self.goal = self.find_optimal_intercept()
            rospy.loginfo(f"Optimal intercept found at x={self.goal.x:.3f}, y={self.goal.y:.3f}")

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