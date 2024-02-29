#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from robot_publisher import Mqtt_Publisher
import base64
import time

class PubImageServer:
    def __init__(self):

        self.sub_img = rospy.Subscriber('/usb_cam/image_raw', Image, self.save_image)
        self.s = rospy.Service('/PubImage', Trigger, self.pub_image)
        self.image = Image()
        self.pub_img = Mqtt_Publisher(node_name='ROBOT')
        rospy.loginfo("PubImageServer start!")

    def save_image(self, msg):
        self.image = msg
    # 处理服务请求的回调函数
    def pub_image(self, req):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(self.image, 'bgr8')
        img_str = cv2.imencode('.jpg', img)[1]
        jpg_as_text = base64.b64encode(img_str).decode('utf-8')
        uri = 'data:image/jpeg;base64,' + jpg_as_text
        self.pub_img.Publish('robot_image', uri)
        time.sleep(0.5)

        # 返回成功响应
        return TriggerResponse(success=True, message="published image successfully.")
