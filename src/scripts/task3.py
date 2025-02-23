#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill
from flyt_task import utils


class Turtle:
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

        # PD - Control parameters
        self.Kp = 15.0
        self.Kd = 3.5

        # Error Term
        self.prev_distance_error = 0.0

        # Trajectory of the circular path (x, y, time)
        self.trajectory = utils.generate_circular_trajectory(self.time, self.radius)

        # Track start time of drawing
        self.start_time = None

        """The turtle spawns at the 0th waypoint at 0 time."""

        # Track next waypoint
        self.current_waypoint = 1
        
        # Initialize goal as the first waypoint
        self.goal = Pose()
        self.goal.x = self.trajectory[1][0]
        self.goal.y = self.trajectory[1][1]

        # Current pose data
        self.current_pose = Pose()

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

    def pose_callback(self, data: Pose):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.

        Publishes the real Pose and the noisy Pose every 5 seconds.

        Args:
            data (Pose): incoming pose data from TurtleSim
        """
        self.current_pose = data

        # Set start time when the first pose callback occurs
        if self.start_time is None:
            self.start_time = rospy.Time.now()

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

            rospy.loginfo(f"Reached Waypoint: {self.current_waypoint}. Time Expected: {time_expected:.2f}. Time Elapsed: {time_elapsed:.2f}")

            # Reset start time upon completion of the circle
            if self.current_waypoint == 0:
                rospy.loginfo(f"Circle completed in {time_elapsed:.2f} seconds!")
                self.start_time = rospy.Time.now()

            else:
                if time_elapsed > time_expected:
                    rospy.logwarn(f"Reached waypoint {self.current_waypoint} late by {time_elapsed-time_expected:.2f}s!")
                    
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
            

if __name__ == "__main__":
    try:
        """Use radius greater 3.0 units, time more than 15 seconds"""
        turtle = Turtle(3.5, 15.0)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
