#!/usr/bin/python
# -*- coding:utf-8 -*-

import rospy
import yaml
import dynamic_reconfigure.client
import rospkg

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('load_yaml_params')

    # 读取YAML文件
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('modify_params')
    print(pkg_path)
    with open(pkg_path + '/scripts/params/narrow_params1.yaml', 'r') as f:
        params1 = yaml.safe_load(f)
    with open(pkg_path + '/scripts/params/narrow_params2.yaml', 'r') as f:
        params2 = yaml.safe_load(f)

    # 加载动态参数
    client1 = dynamic_reconfigure.client.Client("/move_base/TebLocalPlannerROS/")
    for key, value in params1.items():
        print(key, value)
        client1.update_configuration({key : value})
        
    client2 = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer/")
    for key, value in params2.items():
        print(key, value)
        client2.update_configuration({key : value})

    rospy.loginfo('Loaded parameters from params.yaml to parameter server')
