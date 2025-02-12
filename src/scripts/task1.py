#!/usr/bin/python3

'''
Use the default turtle to draw a red 'X' to mark the goal (center).
Kill the default turtle and spawn a second turtle at a random point.
Implement a PID Controller for the second turtle to reach the goal.
'''

import rospy
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute
import random
import math

class Turtle:
    def __init__(self):
        rospy.init_node("turtle", anonymous=True)

        rospy.wait_for_service('kill')
        rospy.wait_for_service('spawn')
        rospy.wait_for_service('turtle1/set_pen')
        rospy.wait_for_service('turtle1/teleport_absolute')

        self.kill = rospy.ServiceProxy('kill', Kill)
        self.spawn = rospy.ServiceProxy('spawn', Spawn)
        self.set_pen = rospy.ServiceProxy('turtle1/set_pen', SetPen)
        self.teleport = rospy.ServiceProxy('turtle1/teleport_absolute', TeleportAbsolute)

    def mark_goal(self):
        try:
            # set pen
            self.set_pen(255, 0, 0, 3, 0)

            # draw the marker
            self.teleport(6, 6, math.pi/4)
            self.teleport(5, 5, math.pi/4)
            self.teleport(5.5, 5.5, math.pi/4)
            self.teleport(6, 5, -math.pi/4)
            self.teleport(5, 6, -math.pi/4)
            self.teleport(5.5, 5.5, -math.pi/4)
            rospy.sleep(0.5)

            # kill the turtle after marking the goal
            rospy.loginfo(f"Goal marked; killing default turtle...")
            self.kill("turtle1")
            rospy.sleep(0.5)

        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to mark goal and/or kill turtle: {e}")

    def spawn_turtle(self):
        x = random.uniform(1, 10)
        y = random.uniform(1, 10)
        theta = random.uniform(0, 2 * math.pi)

        try:
            # spawn a turtle at a random location and log the spawn info
            self.spawn(x, y, theta, "turtle1")
            rospy.loginfo(f"Spawned turtle at x:{x:.2f}, y:{y:.2f}, theta:{theta:.2f}")
            rospy.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn turtle: {e}")

if __name__ == "__main__":
    try:
        turtle = Turtle()
        turtle.mark_goal()
        turtle.spawn_turtle()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
