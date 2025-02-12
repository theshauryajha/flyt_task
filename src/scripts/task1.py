#!/usr/bin/python3

import rospy
from turtlesim.msg import Pose
from turtlesim.srv import Spawn
import random
import math

class Turtle:
    def __init__(self):
        rospy.init_node("turtle", anonymous=True)
        rospy.wait_for_service('spawn')
        self.spawm_service = rospy.ServiceProxy('spawn', Spawn)

    def spawn_turtle(self):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        theta = random.uniform(0, 2 * math.pi)

        try:
            self.spawm_service(x, y, theta, "turtle2")
            rospy.loginfo(f"Spawned turtle at x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn turtle: {e}")

if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.spawn_turtle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
