#!/usr/bin/python
# -*- coding:utf-8 -*-

import random
from geometry_msgs.msg import PoseStamped
from mqtt_robot_service.srv import IsObstacle, IsObstacleRequest
import rospy

def find_quadrilateral_centroid(a, b, c, d):
    """
    This function finds the centroid of a quadrilateral given the coordinates of its four vertices a, b, c, and d.
    """
    x = (a.pose.position.x + b.pose.position.x + c.pose.position.x + d.pose.position.x) / 4
    y = (a.pose.position.y + b.pose.position.y + c.pose.position.y + d.pose.position.y) / 4
    center_point = PoseStamped()
    center_point.pose.position.x = x
    center_point.pose.position.y = y
    return center_point

# 三角形区域生成导航点（优先使用重心，如有障碍物则随机采点）
def generate_random_points_in_triangle(v1, v2, v3):
    pose = PoseStamped()
    # 角度随机
    pose.pose.orientation.z = random.uniform(-1, 1)
    pose.pose.orientation.w = random.uniform(-1, 1)
    
    x = (v1.pose.position.x + v2.pose.position.x + v3.pose.position.x) / 3
    y = (v1.pose.position.y + v2.pose.position.y + v3.pose.position.y) / 3
    # TODO:条件改为判断点所在栅格是否存在障碍物
    client = rospy.ServiceProxy("IsObstacle", IsObstacle)
    client.wait_for_service()
    req = IsObstacleRequest()
    req.x = x
    req.y = y
    res = client.call(req)
    
    while not res:
        # 随机生成两个权重因子
        r1 = random.random()
        r2 = random.random()
        # 根据权重因子计算点的坐标
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = v1.pose.position.x * r1 + v2.pose.position.x * r2 + v3.pose.position.x * (1 - r1 - r2)
        y = v1.pose.position.y * r1 + v2.pose.position.y * r2 + v3.pose.position.y * (1 - r1 - r2)
        req.x = x
        req.y = y
        res = client.call(req)
        
    pose.pose.position.x = x
    pose.pose.position.y = y
    
    return pose

# 根据给出的房间范围生成导航点序列
def get_room_navigation_poses(vertices):


    # 得到四边形重心
    center_point = find_quadrilateral_centroid(vertices[0], vertices[1], vertices[2], vertices[3])

    # 导航点
    nav_poses = []
    # 重心将四边形分割成4个三角形，每个三角形随机生成一个点
    for i in range(8):
        nav_poses.append(generate_random_points_in_triangle(center_point, vertices[i % 4], vertices[(i + 1) % 4]))
    return nav_poses


# # 绘制四边形和随机点
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(A[0]-1,max(A[0], B[0], C[0], D[0])+1)
# ax.set_ylim(A[1]-1, max(A[1], B[1], C[1], D[1])+1)
# # 绘制四边形
# ax.plot([A[0], B[0], C[0], D[0], A[0]], [A[1], B[1], C[1], D[1], A[1]], color='red')
# for point in nav_points:
#     ax.plot(point[0], point[1], 'ro')
# ax.plot(center_point[0], center_point[1], 'bo')
# plt.show()
