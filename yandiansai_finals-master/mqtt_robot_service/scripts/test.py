#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
sys.path.append('../../../devel/lib/python2.7/dist-packages/')

import rospy
import time
import math
from mqtt_robot_service.srv import *
from move_robot import MoveRobot
from mqtt_robot_service.srv import FinalStep, FinalStepResponse
from object_information_msgs.msg import *
from geometry_msgs.msg import Pose
from rosgraph_msgs.msg import Log
from mqtt_robot_service.srv import YoloResult, YoloResultResponse

# demo：没有导航，每隔2s原地旋转60度，检测到人后停下，根据算法调整角度并前进

if __name__ == "__main__":
    
    rospy.init_node("test")
    
    # 准备
    yolo_res = YoloResultResponse()
    yolo_client = rospy.ServiceProxy("/YoloResult", YoloResult)
    yolo_client.wait_for_service()
    robot = MoveRobot()
    
    # 当没检测到人时，一直旋转并检测
    while True:
        # 休眠2s，因为推理结果延迟2s
        print("休眠2s")
        time.sleep(2.0)
        yolo_res = yolo_client.call()
        print("成功获取yolo结果")
        if yolo_res.time_interval < 2.0:
            print(yolo_res)
            break
        # 旋转大约60度
        print("未检测到人，准备旋转")
        robot.rotate(0.8, 2, True)
        time.sleep(2)
    
# 角度矫正
    distance_from_center_rate = (yolo_res.x + (yolo_res.width / 2) - 320) / 320
    rospy.loginfo("get the last yolo result successfully")
    
    # 判断旋转方向,flag = True时向左转
    rotate_flag = False
    if distance_from_center_rate < 0:
        rotate_flag = True
    print("准备矫正角度")
    
    # 根据候选框的水平中心与图像水平中心的距离控制机器人的旋转角度，旋转一次
    
    angular_speed = 0.3
    # rotate_time需要+1s给机器人暖机
    rotate_time = abs(math.atan((distance_from_center_rate / math.sqrt(3))) / angular_speed) + 1.0
    print(rotate_time)
    robot.rotate(angular_speed, rotate_time, rotate_flag)
    time.sleep(rotate_time + 0.5)
    print("rotate end")
    
    # 根据激光雷达的数据控制机器人向前一段距离【通过服务获取小车向前的距离】
    lidar_client = rospy.ServiceProxy("/FinalStep", FinalStep)
    lidar_client.wait_for_service()
    final_step = lidar_client.call()
    linear_speed = 0.2
    move_time = final_step.final_step / linear_speed
    print("move_time:{}".format(move_time))
    robot.move(linear_speed, move_time)
    final_step = lidar_client.call()
    time.sleep(move_time + 4.0)
