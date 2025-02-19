#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Move a TurtleSim turtle in a circular trajectory around the center of the TurtleSim frame.
Provide variables to control radius and speed of the turtle.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill
from math import pi, sqrt, atan2, sin, cos
import numpy as np


class Turtle:
    """
    TurtleSim turtle turtle that implements a P - Controller on the turtle's forward and strafe velocities.
    Generate waypoints for a circular trajectory (at every 1 degree).

    Publishes the real Pose of the turtle as well as the real Pose with a random Gaussian noise, every 5 seconds.
    """
    def __init__(self, radius=3.5, time=3.0):
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

        # P - Control parameter
        self.Kp = 50.0

        # Trajectory of the circular path (x, y, t)
        self.trajectory = self.generate_trajectory()

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

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup publishers for throttled pose
        self.throttled_pub = rospy.Publisher('rt_real_pose', Pose, queue_size=10)
        self.last_published_time = rospy.Time.now()

        # Setup a publisher to publish noisy pose
        self.noisy_pub = rospy.Publisher('rt_noisy_pose', Pose, queue_size=10)
        self.noisy_pose = Pose()

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

        self.draw_circle()

        # Publish the current pose of the turtle every 5 seconds
        if (rospy.Time.now() - self.last_published_time).to_sec() >= 5.0:
            self.throttled_pub.publish(self.current_pose)

            # Add random Gaussian noise (std dev = 10) to the current position
            self.noisy_pose = self.current_pose
            self.noisy_pose.x +=  np.random.normal(0, 10)
            self.noisy_pose.y +=  np.random.normal(0, 10)
            self.noisy_pose.theta += np.random.normal(0, 10)

            self.noisy_pub.publish(self.noisy_pose)

            self.last_published_time = rospy.Time.now()
    
    def generate_trajectory(self):
        """
        Generates the trajectory of the circular path as 3-tuples: (x, y, t) where
        x, y: represent the Cartesian co-ordinates of a waypoint.
        t: represents the time at which the waypoint is reached.

        Returns:
            list of tuples: A list of waypoints in (x, y, t) form
        """
        time_steps = np.linspace(0, self.time, num=360, endpoint=False)
        trajectory = []
        for i in range (360):
            angle = i * (2 * pi / 360)
            x = self.center_x + self.radius * cos(angle)
            y = self.center_y + self.radius * sin(angle)
            t = time_steps[i]

            trajectory.append((x, y, t))
            rospy.loginfo(f"Waypoint {i}: x={x:.3f}, y={y:.3f}, time={t:.3f}")
    
        return trajectory

    def calculate_distance_error(self):
        """
        Calculates the Euclidian distance between the goal and current position of the turtle.
        Returns:
            float: Current distance error
        """
        return sqrt((self.goal.x - self.current_pose.x)**2 + (self.goal.y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        """
        Calculates the smallest angular error between the current heading of the turtle and direction to the goal.
        Returns:
            float: Smallest angle difference (in radians)
        """
        desired_angle = atan2((self.goal.y - self.current_pose.y), (self.goal.x - self.current_pose.x))
        angle_error = desired_angle - self.current_pose.theta

        # Normalise angle
        return self.wrap_angle(angle_error)
    
    def wrap_angle(self, theta):
        """
        Normalizes angle to range [-pi, pi]
        Args:
            theta (float): Angle to be normalized
        Returns:
            float: Normalized angle
        """
        return atan2(sin(theta), cos(theta))
    
    def draw_circle(self):
        """
        Uses the next waypoint as a goal and implements a similar P - Controller as before to move to it.
        Once the waypoint is reached, it updates the goal to the next waypoint.
        """
        # Calculate error
        distance_error = self.calculate_distance_error()
        angle_error = self.calculate_angle_error()

        # P - Control for translation
        target_velocity = self.Kp * distance_error
        
        velocity_magnitude = target_velocity
        velocity_direction = angle_error

        velocity_global = np.array([
            [velocity_magnitude * cos(velocity_direction)],
            [velocity_magnitude * sin(velocity_direction)]
        ])

        # Transform to local frame
        orientation = self.wrap_angle(self.current_pose.theta)
        rotation_matrix = np.array([
            [cos(orientation), sin(orientation)],
            [-sin(orientation), cos(orientation)]
        ])
        velocity_local = rotation_matrix @ velocity_global

        # Create a Twist message
        cmd = Twist()

        # Check if the current waypoint has been reached
        if distance_error < 0.1:
            # Calculate time taken to reach this waypoint
            time_elapsed = (rospy.Time.now() - self.start_time).to_sec()

            # Check if it was reached in time
            time_expected = self.trajectory[self.current_waypoint][2]

            rospy.loginfo(f"Reached Waypoint: {self.current_waypoint}. Time Expected: {time_expected:.2f}. Time Elapsed: {time_elapsed:.2f}")

            # Check if it is the start point
            if self.current_waypoint == 0:
                rospy.loginfo(f"Circle completed in {time_elapsed:.2f} seconds!")
                self.start_time = rospy.Time.now()
            else:
                if time_elapsed > time_expected:
                    rospy.logwarn(f"Reached waypoint {self.current_waypoint} late by {time_elapsed-time_expected:.2f}s!")
                while (rospy.Time.now() - self.start_time).to_sec() <= time_expected:
                    self.cmd_pub.publish(cmd) # Hold turtle still

            self.current_waypoint = (self.current_waypoint + 1) % 360
            self.goal.x = self.trajectory[self.current_waypoint][0]
            self.goal.y = self.trajectory[self.current_waypoint][1]
        

        cmd.linear.x = velocity_local[0].item()
        cmd.linear.y = velocity_local[1].item()

        # Publish the control signals
        self.cmd_pub.publish(cmd)
            

if __name__ == "__main__":
    try:
        turtle = Turtle(3.5, 30)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
