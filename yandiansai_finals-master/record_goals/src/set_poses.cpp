#include "ros/ros.h"
#include "std_msgs/String.h"
#include <fstream>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>

ros::Publisher marker_array_pub;
visualization_msgs::MarkerArray marker_array_poses;
std::ofstream outfile;
std::string record_file;
int cnt = 0;
void cb_record_poses(const geometry_msgs::PoseStamped::ConstPtr& msg){
    outfile.open(record_file, std::ios::binary | std::ios::out | std::ios::app);
    ROS_INFO("goal %d position: %f, %f", cnt, msg->pose.position.x, msg->pose.position.y);
    outfile << msg->pose.position.x << ' ' << msg->pose.position.y <<' '  
            << msg->pose.orientation.z << ' ' << msg->pose.orientation.w << "\n";
    outfile.close();

    //发布markerArray
    visualization_msgs::Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time();
    m.ns = "my_namespace";
    m.id = cnt;
    m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.text = std::to_string(cnt++);
    m.action = visualization_msgs::Marker::ADD;
    m.scale.x = 0.5;
    m.scale.y = 0.5;
    m.scale.z = 0.5;
    m.color.a = 1;
    m.color.r = 1;
    m.color.g = 0;
    m.color.b = 0;
    m.pose = msg->pose;

    marker_array_poses.markers.push_back(m);

    marker_array_pub.publish(marker_array_poses);

}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "set_poses_node");
    ros::NodeHandle n;
    n.getParam("/record_file", record_file);
    
    //初始化目标文件
    outfile.open(record_file);
    outfile.clear();
    outfile.close();

    ros::Subscriber goal_sub = n.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10, cb_record_poses);
    marker_array_pub = n.advertise<visualization_msgs::MarkerArray>("/marker_array_poses", 10);
    
    ros::spin();
    return 0;
}