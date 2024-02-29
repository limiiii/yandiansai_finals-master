#!/usr/bin/python
# -*- coding:utf-8 -*-

from mqtt_robot_service.msg import RoomsNavPoses
from move_base_msgs.msg import MoveBaseActionResult
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

class PubNavgationPoses:
    def __init__(self):
        rospy.init_node("pub_nav_poses_node")
        self.room_sub = rospy.Subscriber("/rooms_nav_poses", RoomsNavPoses, self.pub_poses)
        self.cancel_sub = rospy.Subscriber("/cancel_navigation_flag", Bool, self.cancel_navigation)
        self.status_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.get_status, queue_size=1)
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.r = rospy.Rate(2)
        self.status = 0
        self.cancel_navigation_flag = False

    def cancel_navigation(self, msg):
        self.cancel_navigation_flag = msg.data
    def pub_poses(self, msg):
        for pose in msg.poses:
            if self.cancel_navigation_flag:
                break
            self.status = 0
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose = pose.pose
            self.r.sleep()
            self.pub.publish(goal_pose)
            print("1")
            while not self.cancel_navigation_flag:
                if self.status == 3:
                    break
            print("2")
    
    def get_status(self, data):
        self.status = data.status.status

if __name__ == "__main__":
    pub = PubNavgationPoses()
    print("PubNavgationPoses start!")
    rospy.spin()
