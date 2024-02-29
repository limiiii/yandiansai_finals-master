#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from std_msgs.msg import Bool
from mqtt_robot_service.srv import Stop, StopResponse
import time

class BackupServer:
    def __init__(self):

        self.sub_img = rospy.Subscriber('/if_stop', Bool, self.cb_save_if_stop)
        self.s = rospy.Service('/Stop', Stop, self.stop)
        self.stop_flag = StopResponse()
        rospy.loginfo("BackupServer start!")

    def cb_save_if_stop(self, msg):
        self.stop_flag.stop_flag = msg
    # 处理服务请求的回调函数
    def stop(self, req):
        if self.stop_flag.stop_flag:
            self.stop_flag.stop_flag = False
            return True
        else:
            return False
