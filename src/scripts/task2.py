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
import random
import math

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
        self.Kp_linear = 0.8
        self.Ki_linear = 0.0
        self.Kd_linear = 0.0

        self.Kp_angular = 1.2
        self.Ki_angular = 0.0
        self.Kd_angular = 0.0

        # Error terms
        self.prev_distance_error = 0.0
        self.distance_error_integral = 0.0
        self.prev_angle_error = 0.0
        self.angle_error_integral = 0.0

        # Goal coordinates
        self.goal_x = 5.5
        self.goal_y = 5.5

        # Current pose data
        self.current_pose = Pose()

        # Setup publisher and subscriber for pose and command velocity
        self.pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup publishers to use rqt_mutliplot
        self.distance_error_pub = rospy.Publisher('/turtle2/distance_error', Float64, queue_size=10)
        self.angle_error_pub = rospy.Publisher('/turtle2/angle_error', Float64, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', Pose, queue_size=10)

        # Track current linear and angular velocities
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # Define maximum linear and angular acceleration and deceleration
        self.max_linear_acceleration = 1.0
        self.max_linear_deceleration = 0.2
        self.max_angular_acceleration = 2.0
        self.max_angular_deceleration = 0.8

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
            self.teleport(6, 6, math.pi/4)
            self.teleport(5, 5, math.pi/4)
            self.teleport(5.5, 5.5, math.pi/4)
            self.teleport(6, 5, -math.pi/4)
            self.teleport(5, 6, -math.pi/4)
            self.teleport(5.5, 5.5, -math.pi/4)
            rospy.sleep(0.5)

            # Kill the turtle after marking the goal
            rospy.loginfo("Goal marked; killing default turtle...")
            self.kill("turtle1")
            rospy.sleep(0.5)

        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to mark goal and/or kill turtle: {e}")

    def spawn_turtle(self):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        theta = random.uniform(0, 2 * math.pi)

        try:
            # Spawn a turtle at a random location and log the spawn info
            self.spawn(x, y, theta, "turtle2")
            rospy.loginfo(f"Spawned turtle at x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f}")
            rospy.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn turtle: {e}")

    def calculate_distance_error(self):
        # Euclidian Distance = ((x2-x1)^2 + (y2-y1)^2)^0.5
        return math.sqrt((self.goal_x - self.current_pose.x)**2 + (self.goal_y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        # Slope of line between two points: arctan((y2-y1) / (x2-x1))
        desired_angle = math.atan2((self.goal_y - self.current_pose.y), (self.goal_x - self.current_pose.x))
        angle_error = desired_angle - self.current_pose.theta

        # Normalise angle to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

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

        # Use PID to calculate target velocities
        target_linear_velocity = (self.Kp_linear * distance_error)
        + (self.Kd_linear * distance_error_derivative)
        + (self.Ki_linear * self.distance_error_integral)

        target_angular_velocity = (self.Kp_angular * angle_error)
        + (self.Kd_angular * angle_error_derivative)
        + (self.Ki_angular * self.angle_error_integral)

        # Apply limits to change in linear velocity
        linear_velocity_delta = target_linear_velocity - self.current_linear_velocity
        if linear_velocity_delta > 0: # acceleartion
            max_delta = self.max_linear_acceleration * dt
            linear_velocity_delta = min(linear_velocity_delta, max_delta)
        else: # deceleration
            max_delta = self.max_linear_deceleration * dt
            linear_velocity_delta = max(linear_velocity_delta, -max_delta)
        
        # Apply limits to change in angular velocity
        angular_velocity_delta = target_angular_velocity - self.current_angular_velocity
        if angular_velocity_delta > 0: # acceleartion
            max_delta = self.max_angular_acceleration * dt
            angular_velocity_delta = min(angular_velocity_delta, max_delta)
        else: # deceleration
            max_delta = self.max_angular_deceleration * dt
            angular_velocity_delta = max(angular_velocity_delta, -max_delta)

        # Update current velocities
        self.current_linear_velocity += linear_velocity_delta
        self.current_angular_velocity += angular_velocity_delta

        # Create a Twist message
        cmd_vel = Twist()
        cmd_vel.linear.x = self.current_linear_velocity
        cmd_vel.angular.z = self.current_angular_velocity

        # Publish command velocity and log current pose data
        self.pub.publish(cmd_vel)
        rospy.loginfo(f"Current pose data = x:{self.current_pose.x:.2f}, y:{self.current_pose.y:.2f}, theta:{self.current_pose.theta:.2f}")

        # Update the previous error terms
        self.prev_distance_error = distance_error
        self.prev_angle_error = angle_error

        self.distance_error_pub.publish(Float64(distance_error))
        self.angle_error_pub.publish(Float64(angle_error))

        goal_pose = Pose()
        goal_pose.x = self.goal_x
        goal_pose.y = self.goal_y
        self.goal_pub.publish(goal_pose)

        # Stop at goal and log progress
        if distance_error < 0.01:
            rospy.loginfo(f"Goal reached!")
            #rospy.loginfo(f"Current pose data = x:{self.current_pose.x:.2f}, y:{self.current_pose.y:.2f}, theta:{self.current_pose.theta:.2f}")
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0
            self.pub.publish(cmd_vel)

if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass