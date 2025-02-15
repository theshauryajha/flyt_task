#!/usr/bin/python3

import rospy
from std_msgs.msg import Float64
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill
from math import pi, sqrt, atan2, sin, cos

class Turtle:
    def __init__(self):
        rospy.init_node("turtle", anonymous=True)

        # Setup services
        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        
        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)

        # Start at Pose (x, y, theta) = (1.0, 1.0, 0.0)
        self.kill("turtle1")
        self.spawn(1, 1, 0, "turtle1")

        # PID Controller parameters
        self.Kp_linear = 1.75
        self.Ki_linear = 0.0
        self.Kd_linear = 0.75

        self.Kp_angular = 2.25
        self.Ki_angular = 0.0
        self.Kd_angular = 1.0

        # Error terms
        self.prev_distance_error = 0.0
        self.distance_error_integral = 0.0
        self.prev_angle_error = 0.0
        self.angle_error_integral = 0.0

        # Define waypoints for the lawnmower pattern
        self.waypoints = [(1,1), (10,1), (10,4), (1,4), (1,7), (10,7), (10,10), (1,10)]

        # Track next waypoint
        self.next_waypoint = 0

        # Goal pose to be initialized as first waypoint
        self.goal_pose = Pose()
        self.goal_pose.x, self.goal_pose.y = self.waypoints[self.next_waypoint]

        # Track current pose
        self.current_pose = Pose()
        
        # Setup command publisher and pose subscriber
        self.cmd_pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle1/pose', Pose, self.pose_callback)

        # Track current linear and angular velocities
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # Define maximum linear and angular acceleration and deceleration
        self.max_linear_acceleration = 1.25
        self.max_linear_deceleration = 0.6
        self.max_angular_acceleration = 1.5
        self.max_angular_deceleration = 0.8

        # Track the current time
        self.last_time = rospy.Time.now()

    def pose_callback(self, data):
        self.current_pose = data
        self.move_to_goal()

    def calculate_distance_error(self):
        # Euclidian Distance = ((x2-x1)^2 + (y2-y1)^2)^0.5
        return sqrt((self.goal_pose.x - self.current_pose.x)**2 + (self.goal_pose.y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        # Slope of line between two points: arctan((y2-y1) / (x2-x1))
        desired_angle = atan2((self.goal_pose.y - self.current_pose.y), (self.goal_pose.x - self.current_pose.x))
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

        if abs(angle_error) > (pi / 60):
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

        # Check if a waypoint has been reached
        if distance_error < 0.01:
            self.next_waypoint += 1
            if self.next_waypoint < len(self.waypoints):
                self.goal_pose.x, self.goal_pose.y = self.waypoints[self.next_waypoint]
            else: # The pattern is complete
                rospy.loginfo("Pattern completed!")
                cmd_vel.linear.x = 0
                cmd_vel.angular.z = 0
                self.cmd_pub.publish(cmd_vel)

if __name__ == "__main__":
    try:
        turtle = Turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass