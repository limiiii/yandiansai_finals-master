#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist

class MoveRobot:
    def __init__(self):
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)

    def move(self, linear_speed, duration):
        # 创建Twist消息，设置线速度和角速度
        velocity_msg = Twist()
        velocity_msg.linear.x = linear_speed
        velocity_msg.angular.z = 0.0

        # 发布速度消息，直到指定的时间
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < duration:
            self.velocity_publisher.publish(velocity_msg)
            self.rate.sleep()

        # 停止小车
        velocity_msg.linear.x = 0.0
        velocity_msg.angular.z = 0.0
        self.velocity_publisher.publish(velocity_msg)
    
    
    def rotate(self, angular_speed, duration, flag):
        # 创建Twist消息，设置线速度和角速度
        velocity_msg = Twist()
        velocity_msg.linear.x = 0.0
        if flag == True:
            velocity_msg.angular.z = angular_speed
        else:
            velocity_msg.angular.z = -angular_speed

        # 发布速度消息，直到指定的时间
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < duration:
            self.velocity_publisher.publish(velocity_msg)
            self.rate.sleep()

        # 停止小车
        velocity_msg.linear.x = 0.0
        velocity_msg.angular.z = 0.0
        self.velocity_publisher.publish(velocity_msg)

