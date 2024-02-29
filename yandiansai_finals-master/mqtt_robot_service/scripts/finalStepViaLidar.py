#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from mqtt_robot_service.srv import FinalStep
import math

class FinalStepViaLidar:
    def __init__(self):
        self.sub = rospy.Subscriber('/scan', LaserScan, self.laserScan_callback, queue_size=1)
        self.server = rospy.Service("/FinalStep", FinalStep, self.get_final_step)
        self.laserScan = None
        rospy.loginfo("FinalStepViaLidarServer start!")
        
    def laserScan_callback(self, msg):
        self.laserScan = msg

    def get_final_step(self, req):
        # 获取距离值和角度增量
        ranges = self.laserScan.ranges
        angle_increment = self.laserScan.angle_increment
        
        # 计算角度范围对应的数组下标范围(取前方20度的点云数据)
        # ranges的数据从雷达正前方开始逆时针存放，一共360个数据，每度一个数据
        my_angle = 20
        distances = ranges[0: int(my_angle / 2)] + ranges[int(len(ranges) - my_angle / 2) : len(ranges)]
        # 进行后续处理，若inf数据占3/4，说明贴近目标或雷达检测无效，不动；其他情况找最小的点
        final_distance = 0
        inf_cnt = distances.count(float('inf'))
        filtered_distances = [x for x in distances if x != float('inf')]
        if inf_cnt < int(len(distances)) * 0.9:
            final_distance = min(filtered_distances) * 0.8
        
        # 返回成功响应
        return final_distance
