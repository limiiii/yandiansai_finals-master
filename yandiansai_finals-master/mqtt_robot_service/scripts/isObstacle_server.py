#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from mqtt_robot_service.srv import *
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point


class ObstacleDetectServer:
    def __init__(self):
        self.costmap_sub = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.costmap_callback, queue_size=1)
        self.server = rospy.Service("IsObstacle", IsObstacle, self.doReq)
        self.costmap = None
        rospy.loginfo("ObstacleDetectorServer start!")
        

    def costmap_callback(self, msg):
        self.costmap = msg

    def is_in_obstacle(self, x, y):
        if self.costmap is None:
            return False
        x = int((x - self.costmap.info.origin.position.x) / self.costmap.info.resolution)
        y = int((y - self.costmap.info.origin.position.y) / self.costmap.info.resolution)
        index = y * self.costmap.info.width + x
        if index < 0 or index >= len(self.costmap.data):
            return False
        cost = self.costmap.data[index]
        return (cost != 0)

    def doReq(self, req):
        return self.is_in_obstacle(req.x, req.y)
    
