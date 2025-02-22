#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
from random import uniform
from math import pi, sin, cos
import numpy as np
from flyt_task import utils


class Turtle:
    """
    TurtleSim turtle that implements an external acceleration and deceleartion profile.
    This profile is implemented by limiting how aggressively the turtle can change its velocity.

    Implements a PD - Controller to move to the goal using forward and strafe velocities.
    """
    def __init__(self):
        rospy.init_node("turtle", anonymous=True)

        # Setup services to mark goal, kill default turtle and spawn new turtle
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        rospy.wait_for_service('turtle1/set_pen')
        rospy.wait_for_service('turtle1/teleport_absolute')

        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.set_pen = rospy.ServiceProxy('turtle1/set_pen', SetPen)
        self.teleport = rospy.ServiceProxy('turtle1/teleport_absolute', TeleportAbsolute)

        # PD - Control parameters
        self.Kp = 1.25
        self.Kd = 2.3

        # Error term
        self.prev_distance_error = 0.0

        # Goal pose
        self.goal = Pose()
        self.goal.x = 5.5
        self.goal.y = 5.5

        # Current pose data
        self.current_pose = Pose()

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Publish goal for rqt_multiplot
        self.goal_pub = rospy.Publisher('goal_pose', Pose, queue_size=10)

        # Define maximum acceleration and deceleration
        self.max_acceleration = 1.0
        self.max_deceleration = 1.0

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.
        """
        self.current_pose = data
        self.move_to_goal()

    def mark_goal(self):
        """
        Marks the goal (say the center of the TurtleSim window) with a red cross,
        by using the TurtleSim SetPen and TeleportAbsolute services.
        Kills the default turtle by using the TurtleSim Kill service.
        """
        try:
            # Set pen color -> red
            self.set_pen(255, 0, 0, 3, 0)

            # Draw the marker
            self.teleport(6, 6, pi/4)
            self.teleport(5, 5, pi/4)
            self.teleport(5.5, 5.5, pi/4)
            self.teleport(6, 5, -pi/4)
            self.teleport(5, 6, -pi/4)
            self.teleport(5.5, 5.5, -pi/4)
            rospy.sleep(0.5)

            # Kill the turtle after marking the goal
            rospy.loginfo("Goal marked; killing default turtle...")
            self.kill("turtle1")
            rospy.sleep(0.5)

        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to mark goal and/or kill turtle: {e}")

    def spawn_turtle(self):
        """
        Generates a random pose (x, y, theta).
        Uses the TurtleSim Spawn service to spawn a new turtle at this random pose.
        """
        x = uniform(1, 10)
        y = uniform(1, 10)
        theta = uniform(0, 2 * pi)

        try:
            # Spawn a turtle at a random location and log the spawn info
            self.spawn(x, y, theta, "turtle2")
            rospy.loginfo(f"Spawned turtle at x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f}")
            rospy.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn turtle: {e}")
    
    def move_to_goal(self):
        """
        The target velocity vector is calculated using distance and heading errors.
        The direction of this vector is equivalent to the heading error.
        The magnitude of this vector is calculated as the target velocity obtained by the PD - Controller
        and then constrained by the maximum acceleration / deceleration limits.

        The velocity vector is rotated into the local frame of the turtle and published
        as forward and strafe velocities.
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

        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # PD - Control for targeted magintude of velocity
        target_velocity = (self.Kp * distance_error + self.Kd * distance_error_derivative)

        # Apply limits to change in velocity
        velocity_delta = target_velocity - self.current_pose.linear_velocity
        if velocity_delta > 0: # acceleartion
            max_delta = self.max_acceleration * dt
            velocity_delta = min(velocity_delta, max_delta)
        else: # deceleration
            max_delta = self.max_deceleration * dt
            velocity_delta = max(velocity_delta, -max_delta)

        velocity_magnitude = self.current_pose.linear_velocity + velocity_delta
        velocity_direction = angle_error

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals and log current pose data
        self.cmd_pub.publish(cmd)
        rospy.loginfo(f"Current pose data = x:{self.current_pose.x:.2f}, y:{self.current_pose.y:.2f}, theta:{self.current_pose.theta:.2f}")

        self.goal_pub.publish(self.goal)


if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass