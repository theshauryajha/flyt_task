#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module simulates a chase between a Robber Turtle and a Police Turtle.

The Robber Turtle moves in a circular trajectory as in Goal 3.
The Police Turtle spawns after 10 seconds at a random position and chases the Robber Turtle.
The Police Turtle receives the real Pose of the Robber Turtle every 5 seconds.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen
from random import uniform
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

    This class implements a PD-controlled pursuit behavior with velocity profiling
    for smooth acceleration and deceleration. The Police Turtle spawns at a random
    point in the TurtleSim frame and receives updates of the Robber's position only
    every 5 seconds, simulating limited information pursuit.
    """

    def __init__(self, robber_turtle: RobberTurtle):
        """
        Args:
            robber_turtle (RobberTurtle): Reference to the Robber Turtle instance
        """
        self.robber = robber_turtle

        """Use the TurtleSim Spawn service to spawn Police Turtle at a random position."""
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        spawn_x = uniform(0.5, 10.5)
        spawn_y = uniform(0.5, 10.5)
        self.spawn(spawn_x, spawn_y, 0, "turtle3")

        rospy.loginfo(f"Police Turtle spawned at x: {spawn_x:.3f}, y: {spawn_y:.3f}")

        # PD - Control parameters
        self.Kp = 50.0
        self.Kd = 10.0

        # Error term
        self.prev_distance_error = None

        """
        Initializing the tracking variable for the previous distance error to zero causes the
        first published velocity to be unnaturally large.
        """

        # Current pose of the Police Turtle
        self.current_pose = Pose()

        # Track the last known pose of the Robber Turtle
        self.robber_pose = None

        # Publisher for command velocity
        self.cmd_pub = rospy.Publisher('turtle3/cmd_vel', Twist, queue_size=10)

        # Subscriber for pose
        self.pose_sub = rospy.Subscriber('turtle3/pose', Pose, self.pose_callback)

        # Setup subscriber to get the Pose of the Robber Turtle every 5 seconds
        self.robber_sub = rospy.Subscriber('rt_real_pose', Pose, self.robber_callback)

        # Define acceleration and deceleration limits
        self.acceleration_limit = 5.0
        self.deceleration_limit = 13.5

        # Scale limits to compensate for negligible dt values
        self.acceleration_limit *= 1000
        self.deceleration_limit *= 1000
        
        # Track the current time
        self.last_time = rospy.Time.now()

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
    
    def chase_robber(self):
        """
        Execute the pursuit control loop with velocity profiling.

        This method:
        1. Calculates control errors based on last known target position
        2. Applies PD - control with velocity profiling
        3. Enforces acceleration and deceleration limits
        4. Transforms and publishes velocity commands
        """
        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

        # Calculate error
        distance_error = utils.calculate_distance(self.robber_pose, self.current_pose)
        angle_error = utils.calculate_angle(self.robber_pose, self.current_pose)

        # Handle special case for first callback
        if self.prev_distance_error is None:
            self.prev_distance_error = distance_error
            self.last_time = rospy.Time.now()
            return # Skip this control cycle

        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # PD - Control for targeted magintude of velocity
        target_velocity = (self.Kp * distance_error + self.Kd * distance_error_derivative)

        # Apply limits to change in velocity
        velocity_delta_achieved = utils.limit_velocity_delta(target_velocity,
                                                             self.current_pose.linear_velocity,
                                                             self.acceleration_limit,
                                                             self.deceleration_limit,
                                                             dt)

        velocity_magnitude = self.current_pose.linear_velocity + velocity_delta_achieved
        velocity_direction = angle_error

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals
        self.cmd_pub.publish(cmd)

    def stop(self):
        """
        Stops the Police Turtle by publishing zero velocity.
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