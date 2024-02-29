#!/usr/bin/python
# -*- coding:utf-8 -*-

import paho.mqtt.client as mqtt
import random
from main_process import main_process, main_process_backup
from geometry_msgs.msg import PoseStamped 
from rooms_load import rooms_list
import re

# 连接成功回调
class Mqtt_Subscriber:
    """
        mqtt消息通讯接口
    """

    def __init__(self, central_ip='tricycle.mcurobot.com', port=1883,
                 topic_name='test', callback_func=None,
                 node_name='r329', anonymous=True, timeout=60):
        """
            :param central_ip: Broker的地址
            :param port:  端口号
            :param topic_name: 接收的消息名称
            :param callback_func: 指定回调函数
            :param timeout:  连接延时
            :param node_name: 节点名称
            :param anonymous: 是否同时允许多个节点
        """
        self.topic = topic_name
        self.callback = callback_func
        self.broker_ip = central_ip
        self.broker_port = port
        self.timeout = timeout
        self.connected = False
        self.node_name = node_name + str('_sub')
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
        self.client.on_message = self.default_on_message
        self.client.connect(self.broker_ip, self.broker_port, self.timeout)  # 开始连接
        self.client.subscribe(self.topic)
        self.client.loop_start()  # 开启一个独立的循环通讯线程。

    def default_on_message(self, client, userdata, msg):
        """
            默认回调函数
        """
        msg_str = msg.payload.decode('utf-8')
        print(str(self.topic) + ' ' + msg_str)
        
        # 提取房间号
        room_index = int(re.findall(r'\d+', self.topic)[0])

        # 机器人只响应以下状态
        if msg_str == 'help' or msg_str == 'drink' or msg_str == 'return' or msg_str == 'fall':
            # 机器人收到“return”命令
            if msg_str == 'return':
                print('returning to the robot\'s home...')
                main_process(msg.payload.decode('utf-8'), rooms_list[0])
            # 机器人收到其他应响应命令
            else:
                print('moving to room' + str(room_index) + '...')
                main_process(msg_str, rooms_list[room_index])

    def on_connect(self, client, userdata, flags, rc):
        """
            连接到broker的回调函数
        """
        if rc == 0:
            self.connected = True

        else:
            raise Exception("Failed to connect mqtt server.")
