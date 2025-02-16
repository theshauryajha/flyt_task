#!/usr/bin/python3

'''
Implement the same controller as used in Goal 1 - but with an external acceleration profile.
Implement this acceleration profile by limiting how aggressively the turtle can accelerate or decelerate.
'''

import rospy
from std_msgs.msg import Float64
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
from random import uniform
from math import pi, sqrt, atan2, sin, cos
import numpy as np

class Turtle:
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

        # PID Controller parameters
        self.Kp_linear = 1.25
        self.Ki_linear = 0.0
        self.Kd_linear = 2.3

        # Error terms
        self.prev_distance_error = 0.0
        self.distance_error_integral = 0.0

        # Goal pose
        self.goal_pose = Pose()
        self.goal_pose.x = 5.5
        self.goal_pose.y = 5.5

        # Current pose data
        self.current_pose = Pose()

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Publish goal for rqt_multiplot
        self.goal_pub = rospy.Publisher('goal_pose', Pose, queue_size=10)

        # Define maximum acceleration and deceleration
        self.max_linear_acceleration = 1.0
        self.max_linear_deceleration = 1.0

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data):
        self.current_pose = data
        self.move_to_goal()

    def mark_goal(self):
        try:
            # Set pen
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

    def calculate_distance_error(self):
        # Euclidian Distance = ((x2-x1)^2 + (y2-y1)^2)^0.5
        return sqrt((self.goal_pose.x - self.current_pose.x)**2 + (self.goal_pose.y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        # Slope of line between two points: arctan((y2-y1) / (x2-x1))
        desired_angle = atan2((self.goal_pose.y - self.current_pose.y), (self.goal_pose.x - self.current_pose.x))
        angle_error = desired_angle - self.current_pose.theta

        # Normalise angle
        return self.wrap_angle(angle_error)
    
    def wrap_angle(self, theta):
        return atan2(sin(theta), cos(theta))
    
    def move_to_goal(self):
        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

        # Calculate error
        distance_error = self.calculate_distance_error()
        angle_error = self.calculate_angle_error()

        # Update integral terms
        self.distance_error_integral += distance_error

        # Set derivative terms
        distance_error_derivative = distance_error - self.prev_distance_error

        # PID Control for translation
        target_linear_velocity = (self.Kp_linear * distance_error +
                                self.Kd_linear * distance_error_derivative +
                                self.Ki_linear * self.distance_error_integral)

        # Apply limits to change in linear velocity
        linear_velocity_delta = target_linear_velocity - self.current_pose.linear_velocity
        if linear_velocity_delta > 0: # acceleartion
            max_delta = self.max_linear_acceleration * dt
            linear_velocity_delta = min(linear_velocity_delta, max_delta)
        else: # deceleration
            max_delta = self.max_linear_deceleration * dt
            linear_velocity_delta = max(linear_velocity_delta, -max_delta)

        velocity_magnitude = self.current_pose.linear_velocity + linear_velocity_delta
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

        cmd = Twist()

        cmd.linear.x = velocity_local[0].item()
        cmd.linear.y = velocity_local[1].item()
        self.cmd_pub.publish(cmd)

        self.goal_pub.publish(self.goal_pose)

if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass