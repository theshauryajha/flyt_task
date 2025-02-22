#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Implement the same controller as in Goal 2(a).
Use this controller to draw a grid pattern.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill
from flyt_task import utils


class Turtle:
    """
    TurtleSim turtle that implements an external acceleration and deceleartion profile.
    This profile is implemented by limiting how aggressively the turtle can change its velocity.

    Implements a PD - Controller to move to the next waypoint using forward and/or strafe velocities.
    """
    def __init__(self):
        rospy.init_node("turtle", anonymous=True)

        """Spawn a turtle at a fixed start pose: (1, 1, 0), using the TurtleSim Kill and Spawn services."""
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.kill("turtle1")
        self.spawn(1.0, 1.0, 0.0, "turtle1")

        # PD - Control parameters
        self.Kp = 1.25
        self.Kd = 2.3

        # Error term
        self.prev_distance_error = 0.0

        # Define waypoints for the lawnmower pattern
        self.waypoints = [(1,1), (10,1), (10,4), (1,4), (1,7), (10,7), (10,10), (1,10)]

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

        # Publish the control signals
        self.cmd_pub.publish(cmd)

        # Check if a waypoint has been reached
        if distance_error < 0.01:
            self.next_waypoint += 1
            if self.next_waypoint < len(self.waypoints):
                # If there is another waypoint, update the goal
                self.goal.x, self.goal.y = self.waypoints[self.next_waypoint]
            else:
                # If this is the last waypoint, stop the turtle
                rospy.loginfo("Pattern completed!")
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        turtle = Turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
