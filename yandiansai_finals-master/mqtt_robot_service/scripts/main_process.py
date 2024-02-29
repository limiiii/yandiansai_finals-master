#!/usr/bin/python
# -*- coding:utf-8 -*-
import time
import cv2
import rospy
import math
from geometry_msgs.msg import PoseStamped, Twist
from cv_bridge import CvBridge
from robot_publisher import Mqtt_Publisher
from std_msgs.msg import Bool
import base64
from get_nav_poses import get_room_navigation_poses
from mqtt_robot_service.msg import RoomsNavPoses
from std_srvs.srv import Trigger, TriggerResponse
from mqtt_robot_service.srv import FinalStep, FinalStepResponse, YoloResult, YoloResultResponse, Stop
from move_robot import MoveRobot

def main_process(msg, room_vertices):
    
# 带导航功能版
    r = rospy.Rate(2)
    # 导航情况一：充电座
    if len(room_vertices) == 1:
        pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.pose = room_vertices[0].pose
        r.sleep()
        pub.publish(goal_pose)
        
    # 导航情况二：获得房间的所有导航点
    else:
        # cancel_navigation_flag为False表示执行巡逻，True取消巡逻
        nav_cancel_pub = rospy.Publisher("/cancel_navigation_flag", Bool, queue_size=1)
        nav_cancel_msg = Bool()
        nav_cancel_msg.data = False
        nav_cancel_pub.publish(nav_cancel_msg)
        room_poses = get_room_navigation_poses(room_vertices)
        poses_pub = rospy.Publisher("/rooms_nav_poses", RoomsNavPoses, queue_size=10)
        poses_msg = []
        for pose in room_poses:
            poses_msg.append(pose)
        r.sleep()
        poses_pub.publish(poses_msg)
        # 每隔一段时间请求推理结果【这里还需要写一个服务来保存推理结果】
        yolo_res = YoloResultResponse()
        yolo_client = rospy.ServiceProxy("/YoloResult", YoloResult)
        yolo_client.wait_for_service()
        stop_client = rospy.ServiceProxy("/cancel_navigation", Trigger)
        stop_client.wait_for_service()
        while True:
            yolo_res = yolo_client.call()
            if yolo_res.time_interval < 2.0:
                stop_client.call()
                break
            rospy.sleep(0.1)
        rospy.loginfo("detect person successfully")
        # 当订阅结果出现person时，立刻停止导航和所有动作，每隔2s原地旋转一个角度，检测到人后停下，根据算法调整角度并前进
        
        nav_cancel_msg.data = True
        nav_cancel_pub.publish(nav_cancel_msg)
        
        robot = MoveRobot()
        
        # 当没检测到人时，一直旋转并检测
        while True:
            # 休眠2s，因为推理结果延迟2s
            print("休眠2s")
            time.sleep(2.0)
            yolo_res = yolo_client.call()
            print("成功获取yolo结果")
            if yolo_res.time_interval < 3.0:
                print(yolo_res)
                break
            # 旋转大约60度
            print("未检测到人，准备旋转")
            robot.rotate(0.8, 2, True)
            time.sleep(2)
        
        # 角度矫正功能
        distance_from_center_rate = (yolo_res.x + (yolo_res.width / 2) - 320) / 320
        rospy.loginfo("get the last yolo result successfully")
        # 判断旋转方向,flag = True时向左转
        rotate_flag = False
        if distance_from_center_rate < 0:
            rotate_flag = True
        print("准备矫正角度")
        # 根据候选框的水平中心与图像水平中心的距离控制机器人的旋转角度，旋转一次
        angular_speed = 0.3
        rotate_time = abs(math.atan((distance_from_center_rate / math.sqrt(3))) / angular_speed) + 1.0
        print("rotate time:{}".format(rotate_time))
        robot.rotate(angular_speed, rotate_time, rotate_flag)
        time.sleep(rotate_time + 0.5)
        print("rotate end")
        
        # 如果消息为fall, help则需要机器人找到老人后拍一张照片并上传
        if msg == 'fall' or msg == 'help':
            # 机器人拍照
            pub_img_client = rospy.ServiceProxy("/PubImage", Trigger)
            pub_img_client.wait_for_service()
            pub_img_client.call()
        
        # 根据激光雷达的数据控制机器人向前一段距离【通过服务获取小车向前的距离】
        lidar_client = rospy.ServiceProxy("/FinalStep", FinalStep)
        lidar_client.wait_for_service()
        final_step = lidar_client.call()
        linear_speed = 0.2
        move_time = final_step.final_step / linear_speed
        robot.move(linear_speed, move_time)
        final_step = lidar_client.call()
        
        time.sleep(move_time + 0.5)
        

# 不带导航功能版
        
    # # 每隔2s原地旋转一个角度，检测到人后停下，根据算法调整角度并前进
    # yolo_res = YoloResultResponse()
    # yolo_client = rospy.ServiceProxy("/YoloResult", YoloResult)
    # yolo_client.wait_for_service()
    # robot = MoveRobot()
     
    # # 当没检测到人时，一直旋转并检测
    # while True:
    #     # 休眠2s，因为推理结果延迟2s
    #     # print("休眠2s")
    #     time.sleep(2.0)
    #     yolo_res = yolo_client.call()
    #     print("成功获取yolo结果")
    #     if yolo_res.time_interval < 2.0:
    #         print(yolo_res)
    #         break
    #     # 旋转大约60度
    #     print("未检测到人，准备旋转")
    #     robot.rotate(0.8, 2, True)
    #     time.sleep(2)
        
    # # 角度矫正功能
    # distance_from_center_rate = (yolo_res.x + (yolo_res.width / 2) - 320) / 320
    # rospy.loginfo("get the last yolo result successfully")
    # # 判断旋转方向,flag = True时向左转
    # rotate_flag = False
    # if distance_from_center_rate < 0:
    #     rotate_flag = True
    # print("准备矫正角度")
    # # 根据候选框的水平中心与图像水平中心的距离控制机器人的旋转角度，旋转一次
    # angular_speed = 0.3
    # rotate_angle = abs(math.atan((distance_from_center_rate / math.sqrt(3))))
    # rotate_time = rotate_angle / angular_speed + 1.0
    # print("rotate angle:{}".format(rotate_angle))
    # print("rotate time:{}".format(rotate_time))
    # robot.rotate(angular_speed, rotate_time, rotate_flag)
    # time.sleep(rotate_time + 0.5)
    # print("rotate end")
        
    # # 如果消息为fall, help则需要机器人找到老人后拍一张照片并上传
    # if msg == 'fall' or msg == 'help':
    #     # 机器人拍照
    #     pub_img_client = rospy.ServiceProxy("/PubImage", Trigger)
    #     pub_img_client.wait_for_service()
    #     pub_img_client.call()
        
    # # 根据激光雷达的数据控制机器人向前一段距离【通过服务获取小车向前的距离】
    
    # lidar_client = rospy.ServiceProxy("/FinalStep", FinalStep)
    # lidar_client.wait_for_service()
    # final_step = lidar_client.call()
    # linear_speed = 0.2
    # print("准备前进{}m".format(final_step.final_step))
    # move_time = final_step.final_step / linear_speed
    # robot.move(linear_speed, move_time)
    # final_step = lidar_client.call()
        
    # time.sleep(move_time + 0.5)
    # print("到达老人身边")
    
# 不带导航功能和推理功能的无可奈何版
def main_process_backup(msg, room_vertices):

    robot = MoveRobot()
    print("成功获取yolo结果")
    # 一直旋转直到发出信号后停止
    while True:

        vel_msg = Twist()
        if_stop_client = rospy.ServiceProxy("/Stop", Stop)
        if_stop_client.wait_for_service()
        stop_flag = if_stop_client.call()
        if stop_flag.stop_flag:
            vel_msg.angular.z = 0
            robot.velocity_publisher.publish(vel_msg)
            break
        else:
            vel_msg.angular.z = 0.7
            robot.velocity_publisher.publish(vel_msg)
            time.sleep(0.2)
        
    print("成功检测到人")
    time.sleep(2)
    # 如果消息为fall, help则需要机器人找到老人后拍一张照片并上传
    if msg == 'fall' or msg == 'help':
        # 机器人拍照
        pub_img_client = rospy.ServiceProxy("/PubImage", Trigger)
        pub_img_client.wait_for_service()
        pub_img_client.call()
        
    # 根据激光雷达的数据控制机器人向前一段距离【通过服务获取小车向前的距离】
    
    lidar_client = rospy.ServiceProxy("/FinalStep", FinalStep)
    lidar_client.wait_for_service()
    final_step = lidar_client.call()
    linear_speed = 0.2
    print("准备前进{}m".format(final_step.final_step))
    move_time = final_step.final_step / linear_speed
    robot.move(linear_speed, move_time)
    final_step = lidar_client.call()
        
    time.sleep(move_time + 0.5)
    print("到达老人身边")