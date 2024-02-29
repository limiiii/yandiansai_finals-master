#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../devel/lib/python2.7/dist-packages/')

from object_information_msgs.msg import *
import rospy
from mqtt_robot_service.srv import *
from move_robot import MoveRobot
from mqtt_robot_service.srv import FinalStep, FinalStepResponse
from geometry_msgs.msg import Pose
from mqtt_robot_service.srv import YoloResult, YoloResultResponse
from geometry_msgs.msg import Vector3


class YoloResultServer:
    def __init__(self):
        self.sub = rospy.Subscriber("/objects", Object, self.get_result, queue_size=1)
        self.server = rospy.Service("/YoloResult", YoloResult, self.doReq)
        self.yolo_result = None
        rospy.loginfo("YoloResultServer start!")
        

    def get_result(self, msg):
        self.yolo_result = Object()
        self.yolo_result = msg

    def doReq(self, req):
        res = YoloResultResponse()
        if self.yolo_result == None:
            res.x = 0
            res.width = 0
            res.time_interval = 999.99
        else:
            res.x = self.yolo_result.position.position.x
            res.width = self.yolo_result.size.x
            res.time_interval = abs(self.yolo_result.header.stamp.to_sec() - rospy.Time.now().to_sec())
        
        return res

