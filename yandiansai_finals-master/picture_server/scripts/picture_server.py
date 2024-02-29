#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def image_callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except:
        rospy.logerr("Failed to convert image")
        return

    save_path = "/home/pi/catkin_ws/src/robot/picture_server/picture/fall_"  # 将路径转换为绝对路径

    
    global cnt
    filename = save_path + str(cnt) + ".jpg"  # 生成文件名
    cnt = cnt + 1
    cv2.imwrite(filename, cv_image)  # 保存图像
    rospy.loginfo("Image saved to %s", filename)
        

if __name__ == '__main__':
    cnt = 0
    rospy.init_node('image_saver')
    bridge = CvBridge()
    image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback, queue_size=1)
    print("ready!")
    rospy.spin()
