#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Extension of the Controller implemented in Goal 2(a).
Uses a set of waypoints as goals to draw a grid pattern on
the TurtleSim frame.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill
from flyt_task import utils


class Turtle:
    """
    A controller for TurtleSim turtles using PD - control with velocity profiling.

    This class implements a control system that guides a turtle to a goal position
    using a PD - Controller. The controller includes acceleration and deceleration
    profiles to ensure smooth motion. The resulting velocity vector is decomposed
    into the turtle's local frame for forward and strafe control.

    Uses a set of waypoints arranged in a lawnmower pattern as goals to create
    a grid-like coverage of the workspace.
    """

    def __init__(self):
        rospy.init_node("turtle", anonymous=True)

        """Spawn a turtle at a fixed start pose: (1, 1, 0), using the TurtleSim Kill and Spawn services."""
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.kill("turtle1")
        self.spawn(1.0, 1.0, 0.0, "turtle2")

        # PD - Control parameters
        self.Kp = 1.25
        self.Kd = 2.5

        # Error term
        self.prev_distance_error = 0.0

        """Waypoints for lawnmower grid pattern"""
        self.waypoints = [(10,1), (10,4), (1,4), (1,7), (10,7), (10,10), (1,10)]

        # Track next waypoint
        self.current_waypoint = 0

        # Initialize goal as the first waypoint
        self.goal = Pose()
        self.goal.x, self.goal.y = self.waypoints[self.current_waypoint]

        # Current pose data
        self.current_pose = Pose()

        # Publisher for command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)

        # Subscriber for pose
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Define maximum acceleration and deceleration
        self.max_acceleration = 3.0
        self.max_deceleration = 3.0

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data: Pose):
        """
        Updates current pose from TurtleSim pose message.
        Calls the controller function for the current pose.

        Args:
            data (Pose): incoming pose data from TurtleSim
        """
        self.current_pose = data
        self.draw_pattern()
    
    def draw_pattern(self):
        """
        Uses the next waypoint as a goal and implements a similar PD - Controller as before to move to it.
        Once the waypoint is reached, it updates the goal to the next waypoint.
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
        velocity_delta_achieved = utils.limit_velocity_delta(target_velocity,
                                                             self.current_pose.linear_velocity,
                                                             self.max_acceleration,
                                                             self.max_deceleration,
                                                             dt)

        velocity_magnitude = self.current_pose.linear_velocity + velocity_delta_achieved
        velocity_direction = angle_error

        # Rotate the global velocity vector to the Turtle's local frame
        cmd = utils.rotate_velocity_vector(velocity_magnitude, velocity_direction, self.current_pose.theta)

        # Publish the control signals
        self.cmd_pub.publish(cmd)

        # Check if a waypoint has been reached
        if distance_error < 0.01:
            self.current_waypoint += 1

            if self.current_waypoint < len(self.waypoints):
                rospy.loginfo(f"Waypoint reached: x={self.goal.x: .2f}, y={self.goal.y:.2f}")

                # If there is another waypoint, update the goal
                self.goal.x, self.goal.y = self.waypoints[self.current_waypoint]

            else:
                # If this is the last waypoint, stop the turtle
                rospy.loginfo_once("Pattern completed!")
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        turtle = Turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
