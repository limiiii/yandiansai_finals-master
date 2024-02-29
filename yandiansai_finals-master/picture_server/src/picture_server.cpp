#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "picture_server/image_cmd.h"
#include <sstream>
//Class creation to allow the use of camera callback msg in the service
class PictureServer{
   cv::Mat picture;

public:
    int count = 1;
   //callback to get camera data through "image_pub" topic
   void imageCallback(const sensor_msgs::ImageConstPtr& msg){
      try{
         picture = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
      }
      catch (cv_bridge::Exception& e){
         ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
      }
   }
   // service callback that receives "angle" (int representing image name),
   // "path" (path to save image data) and
   // "cmd" (comand confirming if the camera data
   // should be saved). The service response should return a "result" returning 1
   // if the data was correctly saved
   bool check_and_print(picture_server::image_cmd::Request &req,
                        picture_server::image_cmd::Response &res){
        std::string path = "/home/pi/arebot_ws/src/robot/picture_server/picture/";
      if (req.cmd){
         //image name composed by path (finished with "/")+ capture angle+extension
         std::string im_name = path + std::to_string(count) + ".png";
         count++;
         //checking if the picture has a valid content,
         //otherwise system would failed and stop trying to write the image
         if(!picture.empty()){
            if (!cv::imwrite (im_name, picture)){
               res.result = 0;
               std::cout<<"Image can not be saved as '"<<im_name<<"'\n";
            }else{
               // represent success to save the image
               std::cout<<"Image saved in '"<<im_name<<"'\n";
               res.result = 1;
            }
         }else{
            // represent fail to save the image
            res.result = 0;
            ROS_ERROR("Failed to save image\n");
         }
      }else{
         // represent that server was called, but image was not requested
         res.result = 2;
      }
   }
};

int main(int argc, char **argv)
{
   PictureServer mi;
   ros::init(argc, argv, "Img_Ctrl_server");
   ros::NodeHandle nh;
   image_transport::ImageTransport it(nh);
   ros::ServiceServer service = nh.advertiseService("image_cmd", &PictureServer::check_and_print, &mi);
  image_transport::Subscriber sub = it.subscribe("/usb_cam/image_raw", 1, &PictureServer::imageCallback, &mi);

   ros::spin();
}

