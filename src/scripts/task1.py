#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float64
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
from random import uniform
from math import pi
from flyt_task import utils


class Turtle:
    """
    A controller for TurtleSim turtles using proportional control.

    This class implements a control system that guides a turtle to a goal position
    using two separate P - Controllers: one for position and one for orientation.
    The controller first rotates the turtle to face the goal, then moves it
    forward using velocity commands.
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

        # P - Control parameters
        self.Kp_linear = 3.0
        self.Kp_angular = 2.0

        # Goal pose
        self.goal = Pose()
        self.goal.x = 5.5
        self.goal.y = 5.5

        # Current pose data
        self.current_pose = Pose()

        # Publisher for command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)

        # Subscriber for pose
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Publish goal for rqt_multiplot
        self.goal_pub = rospy.Publisher('goal_pose', Pose, queue_size=10)

    def pose_callback(self, data: Pose):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.

        Args:
            data (Pose): incoming pose data from TurtleSim
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

            marker_points = [
                (6, 6, pi/4),
                (5, 5, pi/4),
                (5.5, 5.5, pi/4),
                (6, 5, -pi/4),
                (5, 6, -pi/4),
                (5.5, 5.5, -pi/4)
            ]

            # Draw the marker
            for x, y, theta in marker_points:
                self.teleport(x, y, theta)
                rospy.sleep(0.1)

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
        Execute the control loop to move the turtle to the goal.

        Implements two P - Controllers:
        1. Angular velocity control to face the goal
        2. Linear velocity control to move to the goal

        This ensures proper orientation of the turtle (threshold = 1degree)
        before moving forward (accuracy = 0.01 units).
        """
        # Calculate error
        distance_error = utils.calculate_distance(self.goal, self.current_pose)
        angle_error = utils.calculate_angle(self.goal, self.current_pose)

        # P - Control for rotation
        angular_velocity = self.Kp_angular * angle_error

        # Create a Twist message
        cmd = Twist()

        # Threshold of 1deg for precise rotation
        if abs(angle_error) > (pi / 180):
            cmd.angular.z = angular_velocity
            cmd.linear.x = 0

        else:
            self.angle_error_integral = 0
            self.prev_angle_error = 0
            cmd.angular.z = 0

            # P - Control for translation
            linear_velocity = self.Kp_linear * distance_error
            cmd.linear.x = linear_velocity

        # Publish control signals and log current pose data
        self.cmd_pub.publish(cmd)
        rospy.loginfo(f"Current pose data = x:{self.current_pose.x:.2f}, y:{self.current_pose.y:.2f}, theta:{self.current_pose.theta:.2f}")

        # Publish goal for plotting
        self.goal_pub.publish(self.goal)

        # Stop at goal and log progress
        # Threshold of 0.01 units for accuracy
        if distance_error < 0.01:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            rospy.loginfo_once(f"Goal reached!")


if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
