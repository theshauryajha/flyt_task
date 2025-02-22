import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from math import sqrt, atan2, sin, cos, pi
import numpy as np

def calculate_distance(goal_pose, current_pose):
    """
    Calculates Euclidian distance error between a Turtle's current pose and a goal position.
    Args:
        goal_pose (Pose): 2D Pose to represent the goal position
        current_pose (Pose): current 2D Pose of the Turtle
    Returns:
            float: Euclidian distance between goal_pose and current_pose
    """
    return sqrt((goal_pose.x - current_pose.x)**2 + (goal_pose.y - current_pose.y)**2)

def calculate_angle(goal_pose, current_pose):
    """
    Calculates the smallest angular error between the current heading of a Turtle and the direction to a goal position.
    Args:
        goal_pose (Pose): 2D Pose to represent the goal position
        current_pose (Pose): current 2D Pose of the Turtle
    Returns:
            float: Smallest angle difference (in radians)
    """
    desired_angle = atan2((goal_pose.y - current_pose.y), (goal_pose.x - current_pose.x))
    angle_error = desired_angle - current_pose.theta

    # Normalise angle
    return wrap_angle(angle_error)

def wrap_angle(theta):
        """
        Normalizes angle to range [-pi, pi].
        Args:
            theta (float): Angle to be normalized
        Returns:
            float: Normalized angle
        """
        return atan2(sin(theta), cos(theta))

def generate_circular_trajectory(time, radius):
    """
    Generates the trajectory of the circular path as 3-tuples: (x, y, t) where
    x, y: represent the Cartesian co-ordinates of a waypoint.
    t: represents the time at which the waypoint needs to be reached.
    
    Args:
        time (float): Total time in which circular trajectory must be completed
        radius (float): Radius of circular trajectory

    Returns:
        list of tuples: A list of waypoints in (x, y, t) form
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

def rotate_velocity_vector(magnitude, direction, current_theta):
    """
    Rotates a velocity vector from a global frame to a Turtle's local frame.
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

def add_random_noise(current_pose):
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