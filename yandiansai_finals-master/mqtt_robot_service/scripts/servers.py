#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../../devel/lib/python2.7/dist-packages/')

import rospy
from cancel_navigation_server import *
from isObstacle_server import *
from finalStepViaLidar import *
from yolo_result_server import *
from pub_image_server import *
from backup_server import *
from backup_publisher import *

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node('servers_node')
    # 带导航版
    server1 = CancelNavigationServer()
    server2 = ObstacleDetectServer()
    server3 = FinalStepViaLidar()
    server4 = YoloResultServer()
    server5 = PubImageServer()
    
    # # 不带导航版
    # server3 = FinalStepViaLidar()
    # server4 = YoloResultServer()
    # server5 = PubImageServer()
    
    # # 不带导航和推理版
    # server3 = FinalStepViaLidar()
    # server5 = PubImageServer()
    # server6 = BackupServer()
    
    
    rospy.spin()
