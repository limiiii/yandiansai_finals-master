#!/usr/bin/python
# -*- coding:utf-8 -*-

from geometry_msgs.msg import PoseStamped

# 设定各房间导航目标，机器人的家保存在goal_list[0]，房间X的目标点保存在goal_list[X]
def load_rooms():
    rooms_list = []
    tmp_points = []
    # 保存每个房间的标记点位
    room_points = []
    with open('/home/pi/catkin_ws/src/robot/record_goals/goal_rooms.txt', 'r') as f:
        for line in f:
            room_point = PoseStamped()
            # 将每行数据按照空格分割成单独的值
            data = line.split()
            # 访问每个值
            room_point.pose.position.x = float(data[0])
            room_point.pose.position.y = float(data[1])
            room_point.pose.orientation.z = float(data[2])
            room_point.pose.orientation.w = float(data[3])
            tmp_points.append(room_point)
    rooms_list = [tmp_points[i:i + 4] for i in range(1, len(tmp_points), 4)]
    tmp = []
    tmp.append(tmp_points[0])
    rooms_list.insert(0, tmp)
    return rooms_list

rooms_list = load_rooms()
