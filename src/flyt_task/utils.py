"""
Utility functions for TurtleSim motion control and trajectory generation.

This module provides helper functions for:
- Computing distances and angles between poses
- Generating circular trajectory
- Converting velocity vector from global frame to Turtle's local frame
- Adding random Gaussian noise to given Pose data
- Implementing acceleration and deceleration and profile
"""

import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from math import sqrt, atan2, sin, cos, pi
import numpy as np

def calculate_distance(goal_pose: Pose, current_pose: Pose) -> float:
    """
    Calculates Euclidian distance error between a Turtle's current pose and a goal position.

    Args:
        goal_pose (Pose): 2D Pose to represent the goal position
        current_pose (Pose): Current 2D Pose of the Turtle

    Returns:
            float: Euclidian distance between goal_pose and current_pose
    """
    return sqrt((goal_pose.x - current_pose.x)**2 + (goal_pose.y - current_pose.y)**2)

def calculate_angle(goal_pose: Pose, current_pose: Pose) -> float:
    """
    Calculates the smallest angular error between the current heading of a Turtle and the direction to a goal position.

    Args:
        goal_pose (Pose): 2D Pose to represent the goal position
        current_pose (Pose): Current 2D Pose of the Turtle

    Returns:
            float: Smallest angle difference (in radians)
    """
    desired_angle = atan2((goal_pose.y - current_pose.y), (goal_pose.x - current_pose.x))
    angle_error = desired_angle - current_pose.theta

    # Normalise angle
    return wrap_angle(angle_error)

def wrap_angle(theta: float) -> float:
        """
        Normalizes angle to range [-pi, pi].

        Args:
            theta (float): Angle to be normalized

        Returns:
            float: Normalized angle
        """
        return atan2(sin(theta), cos(theta))

def generate_circular_trajectory(time: float, radius: float) -> list:
    """
    Generates waypoints for a circular trajectory.
    Creates a circular path centered at the center of the TurtleSim frame (5.5, 5.5)
    with specified radius. The trajectory consists of 360 waypoints (every 1 degree)
    evenly spaced in time.
    
    Args:
        time (float): Total time in which circular trajectory must be completed (seconds)
        radius (float): Radius of circular trajectory (TurtleSim coordinate units)

    Returns:
        list[tuple]: A list of waypoints, each as (x, y, t) where:
            x, y are the Cartesian coordinates of the waypoint
            t is the time at which the Turtle is expected to reach the waypoint
    """
    time_steps = np.linspace(0, time, num=360, endpoint=False)
    trajectory = []

    for i in range (360):
        angle = i * (2 * pi / 360)
        x = 5.5 + radius * cos(angle)
        y = 5.5 + radius * sin(angle)
        t = time_steps[i]

        trajectory.append((x, y, t))

    return trajectory

def rotate_velocity_vector(magnitude: float, direction: float, current_theta: float) -> Twist:
    """
    Uses a rotation matrix to convert a velocity vector from the global frame
    to the Turtle's local frame.

    Args:
        magnitude (float): Magnitude of the velocity vector
        direction (float): Direction of the velocity vector

    Returns:
        Twist: Command velocity in the Turtle's local frame
    """
    velocity_global = np.array([
            [magnitude * cos(direction)],
            [magnitude * sin(direction)]
        ])
    
    orientation = wrap_angle(current_theta)

    rotation_matrix = np.array([
            [cos(orientation), sin(orientation)],
            [-sin(orientation), cos(orientation)]
        ])
    
    velocity_local = rotation_matrix @ velocity_global

    # Create a Twist message
    cmd = Twist()
    cmd.linear.x = velocity_local[0].item()
    cmd.linear.y = velocity_local[1].item()

    return cmd

def add_random_noise(current_pose: Pose) -> Pose:
    """
    Adds random Gaussian noise with standard deviation 10 to given Pose data.

    Args:
        current_pose (Pose): Current Pose of the turtle

    Returns:
        Pose: Pose message with random Gaussian noise added to the original Pose data
    """
    noisy_pose = current_pose
    noisy_pose.x +=  np.random.normal(0, 10)
    noisy_pose.y +=  np.random.normal(0, 10)
    noisy_pose.theta += np.random.normal(0, 10)

    return noisy_pose

def limit_velocity_delta(target_velocity: float,
                         current_velocity: float,
                         max_acceleration: float,
                         max_deceleration: float,
                         dt: float
                         ) -> float:
    """
    Limits the change in velocity for a given time delta based on the acceleration profile.

    Args:
        target_velocity (float): velocity computed from PD - Control loop
        current_velocity (float): current linear velocity of the Turtle
        max_acceleration (float): upper limit on acceleration
        max_deceleration (float): upper limit on deceleration
        dt (float): time step

    Returns:
        float: Change in velocity after applying acceleration or deceleration limit
    """
    velocity_delta_required = target_velocity - current_velocity

    if velocity_delta_required > 0: # acceleartion
        max_delta = max_acceleration * dt
        velocity_delta_achieved = min(velocity_delta_required, max_delta)

    else: # deceleration
        max_delta = -max_deceleration * dt
        velocity_delta_achieved = max(velocity_delta_required, max_delta)

    return velocity_delta_achieved