#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Create a chase simulation with Robber and Police turtles in TurtleSim.
The Robber Turtle moves in a circular trajectory as in Task 3.
The Police Turtle spawns after 10 seconds at a random position and chases the Robber Turtle.
The Police Turtle receives the real pose of the Robber Turtle every 5 seconds.
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, SetPen
from math import pi, sqrt, atan2, sin, cos
import numpy as np
from random import uniform


class RobberTurtle:
    """
    TurtleSim turtle turtle that implements a PD - Controller on the turtle's forward and strafe velocities.
    Generate waypoints for a circular trajectory (at every 1 degree) with variable radius and speed.

    Publishes the real Pose of the turtle as well as the real Pose with a random Gaussian noise, every 5 seconds.
    """
    def __init__(self, radius=3.5, time=15.0):
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

        """Use TurtleSim SetPen service to show the Robber Turtle's trajectory in red."""
        rospy.wait_for_service('turtle2/set_pen')
        self.set_pen = rospy.ServiceProxy('turtle2/set_pen', SetPen)
        self.set_pen(255, 0, 0, 3, 0)

        # PD - Control parameters
        self.Kp = 15.0
        self.Kd = 3.5

        # Error Term
        self.prev_distance_error = 0.0

        # Trajectory of the circular path (x, y, time)
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
        self.is_caught = False

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle2/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle2/pose', Pose, self.pose_callback)

        # Setup publishers for throttled pose
        self.throttled_pub = rospy.Publisher('rt_real_pose', Pose, queue_size=10)
        self.last_published_time = rospy.Time.now()

        # Setup a publisher to publish noisy pose
        self.noisy_pub = rospy.Publisher('rt_noisy_pose', Pose, queue_size=10)
        self.noisy_pose = Pose()

        # Track the current time
        self.last_time = rospy.Time.now()

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

        if not self.is_caught:
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
        t: represents the time at which the waypoint needs to be reached.

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
        Uses the next waypoint as a goal and implements a PD - Controller to move to it.
        If the turtle reaches a waypoint earlier than expected, it will wait till it is actually expected to be there.
        """
        # Calculate error
        distance_error = self.calculate_distance_error()
        angle_error = self.calculate_angle_error()
        
        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # P - Control for translation
        target_velocity = self.Kp * distance_error + self.Kd * distance_error_derivative
        
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
            # Calculate time expected as well as time actually taken to reach this waypoint
            time_elapsed = (rospy.Time.now() - self.start_time).to_sec()
            time_expected = self.trajectory[self.current_waypoint][2]

            # Reset start time upon completion of the circle
            if self.current_waypoint == 0:
                self.start_time = rospy.Time.now()
            else:
                while (rospy.Time.now() - self.start_time).to_sec() <= time_expected:
                    # If the turtle is early to this waypoint, hold it still till the time it is actually expected to be there
                    self.cmd_pub.publish(cmd) # Hold turtle still

            # Update waypoint
            self.current_waypoint = (self.current_waypoint + 1) % 360
            self.goal.x = self.trajectory[self.current_waypoint][0]
            self.goal.y = self.trajectory[self.current_waypoint][1]
        
        cmd.linear.x = velocity_local[0].item()
        cmd.linear.y = velocity_local[1].item()

        # Publish the control signals
        self.cmd_pub.publish(cmd)

    def stop(self):
        """Stops the Robber Turtle by setting is_caught flag and publishing zero velocity."""
        self.is_caught = True
        cmd = Twist()
        self.cmd_pub.publish(cmd)


class PoliceTurtle:
    """
    TurtleSim turtle that implements a PD - Controller with an external
    acceleration / deceleration profile, to chase the Robber Turtle.

    Spawns at a random position and receives the Robber Turtle's pose every 5 seconds.
    """
    def __init__(self, robber_turtle):
        """
        Args:
            robber_turtle (RobberTurtle): Reference to the Robber Turtle instance
        """
        self.robber = robber_turtle

        """Use the TurtleSim Spawn service to spawn Police turtle at a random position."""
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        spawn_x = uniform(0.5, 10.5)
        spawn_y = uniform(0.5, 10.5)
        self.spawn(spawn_x, spawn_y, 0, "turtle3")

        # PD - Control parameters
        self.Kp = 50.0
        self.Kd = 10.0

        # Error term
        self.prev_distance_error = None
        """
        Initializing the tracking variable for the previous distance error as 0
        causes the first published velocity to be unnaturally large.
        """

        # Current pose of the Police Turtle
        self.current_pose = Pose()

        # Track the last known pose of the Robber Turtle
        self.robber_pose = None

        # Setup publisher and subscriber for pose and command velocity
        self.cmd_pub = rospy.Publisher('turtle3/cmd_vel', Twist, queue_size=10)
        self.pose_sub = rospy.Subscriber('turtle3/pose', Pose, self.pose_callback)

        # Setup subscriber to get the Pose of the Robber Turtle every 5 seconds
        self.robber_sub = rospy.Subscriber('rt_real_pose', Pose, self.robber_callback)

        # Define acceleration and deceleration limits
        self.acceleration_limit = 4.75
        self.deceleration_limit = 13.5

        # Scale limits to compensate for negligible dt values
        self.acceleration_limit *= 1000
        self.deceleration_limit *= 1000

        # Track current linear velocity
        self.current_linear_velocity = 0.0
        
        # Track the current time
        self.last_time = rospy.Time.now()

    def robber_callback(self, data):
        """Updates the last known position of the Robber Turtle (every 5 seconds)."""
        self.robber_pose = data

    def pose_callback(self, data):
        """Updates current pose of the Police Turtle and checks if the Robber Turtle is caught."""
        self.current_pose = data
        self.current_linear_velocity = self.current_pose.linear_velocity

        if not self.robber.is_caught and self.robber_pose is not None:
            self.chase_robber()

            # Calculate Euclidian distance between Police Turtle and actual Pose of the robber turtle
            distance = sqrt((self.current_pose.x - self.robber.current_pose.x)** 2 +
                            (self.current_pose.y - self.robber.current_pose.y)**2)
            
            """Stop both turtles when the Robber Turtle is caught (tolerance = radius * 0.1)"""
            if distance <= 0.4:
                self.stop()
                self.robber.stop()
                rospy.loginfo("Robber Turtle caught!")

    def calculate_distance_error(self):
        """
        Calculates the Euclidian distance between the Police Turtle and the last known position of the Robber Turtle.
        
        Returns:
            float: Current distance error
        """
        return sqrt((self.robber_pose.x - self.current_pose.x)**2 + (self.robber_pose.y - self.current_pose.y)**2)
    
    def calculate_angle_error(self):
        """
        Calculates the smallest angular error between the current heading of the Police Turtle and the direction to the Robber Turtle.
        Returns:
            float: Smallest angle difference (in radians)
        """
        desired_angle = atan2((self.robber_pose.y - self.current_pose.y), (self.robber_pose.x - self.current_pose.x))
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
    
    def chase_robber(self):
        """
        Uses PD - control to chase the Robber Turtle based on its last known position.
        Applies acceleration and deceleration limits.
        """
        # Calculate time delta
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        if dt == 0:
            return

        #rospy.loginfo(f"Maximum Velocity Incrase: {self.acceleration_limit * dt}")
        #rospy.loginfo(f"Maximum Velocity Decrease: {-self.deceleration_limit * dt}")

        # Calculate error
        distance_error = self.calculate_distance_error()
        angle_error = self.calculate_angle_error()

        # Handle special case for first callback
        if self.prev_distance_error is None:
            self.prev_distance_error = distance_error
            self.current_linear_velocity = 0.0
            self.last_time = rospy.Time.now()
            return # Skip this control cycle

        # Set derivative term
        distance_error_derivative = distance_error - self.prev_distance_error

        # PD - Control for targeted magintude of velocity
        target_velocity = (self.Kp * distance_error + self.Kd * distance_error_derivative)

        # Apply limits to change in velocity
        velocity_delta_required = target_velocity - self.current_linear_velocity
        #rospy.loginfo(f"Required Velocity Delta: {velocity_delta_required}")
        if velocity_delta_required > 0: # acceleartion
            max_delta = self.acceleration_limit * dt
            #if velocity_delta_required > max_delta:
                #rospy.logwarn("Capping Acceleration!")
            velocity_delta_achieved = min(velocity_delta_required, max_delta)
        else: # deceleration
            max_delta = -self.deceleration_limit * dt
            #if velocity_delta_required < max_delta:
                #rospy.logwarn("Capping Deceleration!")
            velocity_delta_achieved = max(velocity_delta_required, max_delta)
        #rospy.loginfo(f"Achieved Velocity Delta: {velocity_delta_achieved}\n\n")

        velocity_magnitude = self.current_linear_velocity + velocity_delta_achieved
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

    def stop(self):
        """
        Stops the Police Turtle by publishing zero velocity
        Sets the is_caught flag of the Robber Turtle.
        """
        self.robber.is_caught = True
        cmd = Twist()
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    try:
        robber_turtle = RobberTurtle(4.0, 15.0)
        rospy.sleep(10)
        police_turtle = PoliceTurtle(robber_turtle)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass