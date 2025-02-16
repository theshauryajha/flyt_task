#!/usr/bin/python3

'''
Implement the same PID Controller as in Goal 1.
Define maximum acceleration and deceleration to implement a deceleration profile.
'''

import rospy
from std_msgs.msg import Float64
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
from random import uniform
from math import pi, sqrt, atan2, sin, cos

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
        self.Kp_linear = 2.0
        self.Ki_linear = 0.0
        self.Kd_linear = 0.0

        self.Kp_angular = 1.0
        self.Ki_angular = 0.0
        self.Kd_angular = 0.0

        # Error terms
        self.prev_distance_error = 0.0
        self.distance_error_integral = 0.0
        self.prev_angle_error = 0.0
        self.angle_error_integral = 0.0

        # Goal pose
        self.goal = Pose()
        self.goal.x = 5.5
        self.goal.y = 5.5

        # Current pose data
        self.current_pose = Pose()

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup publishers to use rqt_mutliplot
        self.distance_error_pub = rospy.Publisher('/turtle2/distance_error', Float64, queue_size=10)
        self.angle_error_pub = rospy.Publisher('/turtle2/angle_error', Float64, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', Pose, queue_size=10)

        # Track current linear and angular velocities
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # Define maximum linear and angular acceleration and deceleration
        self.max_linear_acceleration = 1.0
        self.max_linear_deceleration = 0.1
        self.max_angular_acceleration = 2.0
        self.max_angular_deceleration = 0.5

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
        return sqrt((self.goal.x - self.current_pose.x)**2 + (self.goal.y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        # Slope of line between two points: arctan((y2-y1) / (x2-x1))
        desired_angle = atan2((self.goal.y - self.current_pose.y), (self.goal.x - self.current_pose.x))
        angle_error = desired_angle - self.current_pose.theta

        # Normalise angle
        angle_error = atan2(sin(angle_error), cos(angle_error))
        return angle_error
    
    def move_to_goal(self):
        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

        distance_error = self.calculate_distance_error()
        angle_error = self.calculate_angle_error()

        # Update integral terms
        self.distance_error_integral += distance_error
        self.angle_error_integral += angle_error

        # Set derivative terms
        distance_error_derivative = distance_error - self.prev_distance_error
        angle_error_derivative = angle_error - self.prev_angle_error

        # PID Control for rotation
        target_angular_velocity = (self.Kp_angular * angle_error +
                                   self.Kd_angular * angle_error_derivative +
                                   self.Ki_angular * self.angle_error_integral)

        # Apply limits to change in angular velocity
        angular_velocity_delta = target_angular_velocity - self.current_angular_velocity
        if angular_velocity_delta > 0: # acceleartion
            max_delta = self.max_angular_acceleration * dt
            angular_velocity_delta = min(angular_velocity_delta, max_delta)
        else: # deceleration
            max_delta = self.max_angular_deceleration * dt
            angular_velocity_delta = max(angular_velocity_delta, -max_delta)

        self.current_angular_velocity += angular_velocity_delta

        cmd_vel = Twist()

        if abs(angle_error) > (pi / 90):
            # First, rotate the turtle to face the goal
            cmd_vel.angular.z = self.current_angular_velocity
            cmd_vel.linear.x = 0
        else:
            # When facing the goal, reset angle error and move forward only
            self.angle_error_integral = 0
            self.prev_angle_error = 0
            cmd_vel.angular.z = 0

            # PID Control for translation
            target_linear_velocity = (self.Kp_linear * distance_error +
                                      self.Kd_linear * distance_error_derivative +
                                      self.Ki_linear * self.distance_error_integral)

            # Apply limits to change in linear velocity
            linear_velocity_delta = target_linear_velocity - self.current_linear_velocity
            if linear_velocity_delta > 0: # acceleartion
                max_delta = self.max_linear_acceleration * dt
                linear_velocity_delta = min(linear_velocity_delta, max_delta)
            else: # deceleration
                max_delta = self.max_linear_deceleration * dt
                linear_velocity_delta = max(linear_velocity_delta, -max_delta)

            self.current_linear_velocity += linear_velocity_delta

            cmd_vel.linear.x = self.current_linear_velocity

        self.cmd_pub.publish(cmd_vel)

        # Update the previous error terms
        self.prev_distance_error = distance_error
        self.prev_angle_error = angle_error

        # Plot errors and goal
        self.distance_error_pub.publish(Float64(distance_error))
        self.angle_error_pub.publish(Float64(angle_error))
        self.goal_pub.publish(self.goal)

        # Stop at goal and log progress
        if distance_error < 0.01:
            rospy.loginfo(f"Goal reached!")
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0
            self.cmd_pub.publish(cmd_vel)


if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
