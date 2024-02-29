#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from std_msgs.msg import Bool

class IfStopPublisher:
    def __init__(self):

        self.pub = rospy.Publisher("/if_stop", Bool, queue_size=1)
        rospy.sleep(1)
        msg = False
        self.pub.publish(msg)
        rospy.loginfo("BackupServer start!")