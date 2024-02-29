#!/usr/bin/python
# -*- coding:utf-8 -*-

import paho.mqtt.client as mqtt
import threading
import random
import time
import cv2
import numpy as np

# 连接成功回调
class Mqtt_Publisher:
    """
        mqtt消息通讯接口
    """

    def __init__(self, central_ip='tricycle.mcurobot.com', port=1883, node_name='r329',anonymous='True', timeout=60):
        """
        :param central_ip: Broker的地址
        :param port:  端口号
        :param timeout:  连接延时
        :param node_name: 节点名称
        """
        self.broker_ip = central_ip
        self.broker_port = port
        self.timeout = timeout
        self.connected = False
        self.node_name = node_name + str('_pub')
        if anonymous:
            self.node_name = self.node_name + str('_') + str(random.randint(10000, 99999))
        self.Start()

    def Start(self):
        """
        开启publisher
        :return:
        """
        self.client = mqtt.Client(self.node_name)  # 创建客户端
        self.client.on_connect = self.on_connect  # 指定回调函数
        self.client.connect(self.broker_ip, self.broker_port, self.timeout)  # 开始连接
        self.client.loop_start()  # 开启一个独立的循环通讯线程。

    def Publish(self, topic, payload, qos=2, retain=False):
        """
            发送一个mqtt消息
            :param topic: 消息名称，string类型
            :param payload: 消息内容，string类型
            :param qos: 消息等级
            :retain: 状态机消息
            :return:
        """
        if self.connected:
            return self.client.publish(topic, payload=payload, qos=qos, retain=retain)
        else:
            raise Exception("mqtt server not connected! you may use .Start() function to connect to server firstly.")

    '''
                回调函数
    '''

    def on_connect(self, client, userdata, flags, rc):
        """
            连接到broker的回调函数
        """
        if rc == 0:
            self.connected = True

        else:
            raise Exception("Failed to connect mqtt server.")
