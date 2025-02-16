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
    def __init__(self, radius=2.5, speed=1.0):
        """
        Args:
            radius (float): Radius of the circular trajectory
            speed (float): Scaling factor for revolution speed
        """

        rospy.init_node("turtle", anonymous=True)
        self.radius = radius
        self.speed = speed

        """Use TurtleSim Kill and Spawn services to start at a point on a circle of given radius."""
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.kill("turtle1")

        center_x, center_y = 5.5, 5.5 # Center of TurtleSim frame
        
        # Spawn at (center_x + radius, center_y)
        spawn_x, spawn_y = center_x + self.radius, center_y
        self.spawn(spawn_x, spawn_y, 0, "turtle1")

        # PD - Control parameter
        self.Kp = self.speed * 10.0

        # Generate circular waypoints
        self.waypoints = []
        for i in range(360):
            angle = 2 * pi * i / 360
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            self.waypoints.append((x, y))
        
        # Add the first point again to close the circle
        self.waypoints.append(self.waypoints[0])

        # Track next waypoint
        self.next_waypoint = 0

        # Goal pose to be initialized as first waypoint
        self.goal = Pose()
        self.goal.x, self.goal.y = self.waypoints[self.next_waypoint]
        
        # Current pose data
        self.current_pose = Pose()

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle1/pose', Pose, self.pose_callback)

        # Setup publishers for throttled pose
        self.throttled_pub = rospy.Publisher('rt_real_pose', Pose, queue_size=10)
        self.last_published_time = rospy.Time.now()

        # Setup a publisher to publish noisy pose
        self.noisy_pub = rospy.Publisher('rt_noisy_pose', Pose, queue_size=10)
        self.noisy_pose = Pose()

        # Define maximum acceleration and deceleration
        self.max_acceleration = 10.0
        self.max_deceleration = 10.0

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.

        Publishes the real Pose and the noisy Pose every 5 seconds.
        """
        self.current_pose = data
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

        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

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
        cmd.linear.x = velocity_local[0].item()
        cmd.linear.y = velocity_local[1].item()

        # Publish the control signals
        self.cmd_pub.publish(cmd)

        # Check if a waypoint has been reached
        if distance_error < 0.1:
            self.next_waypoint += 1
            if self.next_waypoint < len(self.waypoints):
                self.goal.x, self.goal.y = self.waypoints[self.next_waypoint]
            else: # Circle completed, repeat
                rospy.loginfo("Circle complete!")
                self.next_waypoint = 0


if __name__ == "__main__":
    try:
        turtle = Turtle(3.0, 8.0)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass