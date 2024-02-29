#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
import yaml
import dynamic_reconfigure.client
import rospkg
import tf2_ros
import math
from geometry_msgs.msg import PoseStamped

# 全局路径
pkg_path = ''
narrow_params = []
wide_params = []
modify_poses = []

# 读取参数等数据
def loadData():
    # 找功能包路径
    rospack = rospkg.RosPack()
    global pkg_path 
    pkg_path = rospack.get_path('modify_params')
    # 读取YAML文件
    global narrow_params, wide_params
    with open(pkg_path + '/params/narrow_params1.yaml', 'r') as f:
        narrow_params.append(yaml.safe_load(f))
    with open(pkg_path + '/params/narrow_params2.yaml', 'r') as f:
        narrow_params.append(yaml.safe_load(f))
    with open(pkg_path + '/params/wide_params1.yaml', 'r') as f:
        wide_params.append(yaml.safe_load(f))
    with open(pkg_path + '/params/wide_params2.yaml', 'r') as f:
        wide_params.append(yaml.safe_load(f))
    # 读取修改参数的pose
    global modify_poses
    with open(pkg_path + '/modify_poses.txt', 'r') as f:
        for line in f:
            pose = PoseStamped()
            # 将每行数据按照空格分割成单独的值
            data = line.split()
            # 访问每个值
            pose.pose.position.x = float(data[0])
            pose.pose.position.y = float(data[1])
            pose.pose.orientation.z = float(data[2])
            pose.pose.orientation.w = float(data[3])
            modify_poses.append(pose)
    rospy.loginfo('load successfully!')

# 修改动态参数
def modifyParams(flag):
    global narrow_params, wide_params
    if flag: 
        client1_narrow = dynamic_reconfigure.client.Client("/move_base/TebLocalPlannerROS/")
        for key, value in narrow_params[0].items():
            client1_narrow.update_configuration({key : value})
        client2_narrow = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer/")
        for key, value in narrow_params[1].items():
            client2_narrow.update_configuration({key : value})
        rospy.loginfo('Now params: narrow')
    else:
        client1_wide = dynamic_reconfigure.client.Client("/move_base/TebLocalPlannerROS/")
        for key, value in wide_params[0].items():
            client1_wide.update_configuration({key : value})
        client2_wide = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer/")
        for key, value in wide_params[1].items():
            client2_wide.update_configuration({key : value})
        rospy.loginfo('Now params: wide')

# 监听小车pose
def getRobotPose():
    transform = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(), rospy.Duration(2.0))
    robot_pose = PoseStamped()
    robot_pose.header = transform.header
    robot_pose.pose.position.x = transform.transform.translation.x
    robot_pose.pose.position.y = transform.transform.translation.y
    robot_pose.pose.orientation.z = transform.transform.rotation.z
    robot_pose.pose.orientation.w = transform.transform.rotation.w
    #rospy.loginfo("pose: ({:f}, {:f})".format(robot_pose.pose.position.x, robot_pose.pose.position.y))
    return robot_pose

# 计算距离和角度差
def distance_and_angle_difference(pose1, pose2):
    # 计算距离
    distance = math.sqrt((pose1.pose.position.x - pose2.pose.position.x)**2 +
                         (pose1.pose.position.y - pose2.pose.position.y)**2)

    # 计算两个点之间的角度差(度)
    q1 = pose1.pose.orientation
    q2 = pose2.pose.orientation
    angle_difference = math.degrees(2 * math.acos(abs(q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z)))

    return distance, angle_difference

# 判断是否需要修改参数
def determineMod(robot_pose, modify_poses, flag):
    for modify_pose in modify_poses:
        distance, angle_difference = distance_and_angle_difference(robot_pose, modify_pose)
        if flag:
            if distance <= 0.1 and angle_difference > 130:
                rospy.loginfo("准备离开狭窄路段")
                flag = False
        else:
            if distance <= 0.1 and angle_difference < 50:
                rospy.loginfo("准备进入狭窄路段")
                flag = True
    
    return flag

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('modify_params_node')
    
    # 读取参数等数据
    loadData()
    
    # 前置变量，实例准备
    rate = rospy.Rate(2)
    flag_if_narrow = False
    pre_flag = False
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    # 持续监听小车pose，如果准备进入狭窄路段，则flag置True，反之置False
    while not rospy.is_shutdown():
        robot_pose = getRobotPose()
        # 如果flag == false， 则监听是否靠近pose的正方向；反之监听是否靠近pose的负方向
        flag_if_narrow = determineMod(robot_pose, modify_poses, flag_if_narrow)
        # 如果flag发生了变化，则修改参数
        if flag_if_narrow != pre_flag:
            modifyParams(flag_if_narrow)
        pre_flag = flag_if_narrow
        
        rate.sleep()
    

    
