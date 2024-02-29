#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from geometry_msgs.msg import Twist

class CancelNavigationServer:
    def __init__(self):

        self.publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # 创建服务
        self.s = rospy.Service('cancel_navigation', Trigger, self.handle_cancel_navigation)
        
        # 初始化MoveBaseAction客户端
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        rospy.loginfo("CancelNavigationServer start!")

    # 处理服务请求的回调函数
    def handle_cancel_navigation(self, req):
        # 取消当前的导航任务
        self.client.cancel_all_goals()
        # 发布停止指令
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)

        # 返回成功响应
        return TriggerResponse(success=True, message="Navigation canceled.")
