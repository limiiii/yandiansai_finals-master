#!/usr/bin/python
# -*- coding:utf-8 -*-

from robot_subscriber import Mqtt_Subscriber
import rospy

if __name__ == '__main__':
    print("------START SUBSCRIBING!------")
    
    rospy.init_node("robot_nav_photo")
    
    
    # 订阅所有发布者
    p1 = Mqtt_Subscriber(node_name='ROOM1', topic_name='room1_sr')
    p2 = Mqtt_Subscriber(node_name='ROOM1', topic_name='room1_fd')
    p3 = Mqtt_Subscriber(node_name='ROOM2', topic_name='room2_sr')
    p4 = Mqtt_Subscriber(node_name='ROOM2', topic_name='room2_fd')
    p5 = Mqtt_Subscriber(node_name='ROOM3', topic_name='room3_sr')
    p6 = Mqtt_Subscriber(node_name='ROOM3', topic_name='room3_fd')
    
    # while not p1.connected:
    #     pass
    rospy.spin()
