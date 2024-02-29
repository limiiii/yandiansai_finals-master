// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "cpu.h"
#include "gpu.h"
#include "object_information_msgs/Object.h"

ncnn::Net yolov3;

object_information_msgs::Object objMsg;
ros::Publisher obj_pub;
image_transport::Publisher image_pub;
std::vector<Object> objects;
cv_bridge::CvImagePtr cv_ptr;
sensor_msgs::ImagePtr image_msg;
bool display_output;
bool enable_gpu;
int thread;

int target_size = 352;
float prob_threshold = 0.5f;
float nms_threshold = 0.45f;

static int detect_yolov3(const cv::Mat& bgr, std::vector<Object>& objects)
{

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(thread); //bd add this line
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static const char* class_names[] = {"background",
                                    "aeroplane", "bicycle", "bird", "boat",
                                    "bottle", "bus", "car", "cat", "chair",
                                    "cow", "diningtable", "dog", "horse",
                                    "motorbike", "person", "pottedplant",
                                    "sheep", "sofa", "train", "tvmonitor"
                                    };      
static const unsigned char colors[19][3] = {
    {54, 67, 244},
    {99, 30, 233},
    {176, 39, 156},
    {183, 58, 103},
    {181, 81, 63},
    {243, 150, 33},
    {244, 169, 3},
    {212, 188, 0},
    {136, 150, 0},
    {80, 175, 76},
    {74, 195, 139},
    {57, 220, 205},
    {59, 235, 255},
    {7, 193, 255},
    {0, 152, 255},
    {34, 87, 255},
    {72, 85, 121},
    {158, 158, 158},
    {139, 125, 96}
};                                                                  
static cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
  int color_index = 0;
  cv::Mat image = bgr.clone();

  for (size_t i = 0; i < objects.size(); i++)
  {
      const Object& obj = objects[i];

      const unsigned char* color = colors[color_index % 19];
      color_index++;
      cv::Scalar cc(color[0], color[1], color[2]);
      cv::rectangle(image, obj.rect, cc, 2);

      char text[256];
      sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

      int baseLine = 0;
      cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

      int x = obj.rect.x;
      int y = obj.rect.y - label_size.height - baseLine;
      if (y < 0)
          y = 0;
      if (x + label_size.width > image.cols)
          x = image.cols - label_size.width;

      cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

      cv::putText(image, text, cv::Point(x, y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
  return image;

}

float fps = 0.0;
uint64 detect_sequence = 0;
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try {
    ros::Time current_time = ros::Time::now();
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    detect_yolov3(cv_ptr->image, objects);
    if(0 == detect_sequence)
      fps = 1.0/(ros::Time::now() - current_time).toSec();
    else
      fps = 0.9*fps + (1.0/(ros::Time::now() - current_time).toSec())*0.1;
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (obj.prob > prob_threshold)
        {
          ROS_INFO("%s = %.5f at %.2f %.2f %.2f x %.2f", class_names[obj.label], obj.prob,
          obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
          objMsg.header.seq++;
          objMsg.header.stamp = msg->header.stamp;
          objMsg.probability = obj.prob;
          objMsg.label = class_names[obj.label];
          objMsg.object_total = objects.size();
          objMsg.object_sequence = i+1;
          objMsg.detect_sequence = msg->header.seq;
          objMsg.position.position.x = obj.rect.x;
          objMsg.position.position.y = obj.rect.y;
          objMsg.size.x = obj.rect.width;
          objMsg.size.y = obj.rect.height;
          obj_pub.publish(objMsg);
        }
    }
    if (display_output) {
      cv::Mat out_image ;
      out_image = draw_objects(cv_ptr->image, objects);
      char text[32];
      sprintf(text, "YOLO V3 FPS %.03f", fps);
      cv::putText(out_image, text, cv::Point(20, 20),cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0));      
      image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
      image_pub.publish(image_msg);
    }
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "yolov3_node"); /**/
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  int powersave;
  nhLocal.param("powersave", powersave, 0);   
  nhLocal.param("thread", thread, 2);   
  nhLocal.param("gpu_device", gpu_device, 0);
  nhLocal.param("enable_gpu", enable_gpu, false);
  nhLocal.param("target_size", target_size, 352);
  nhLocal.param("prob_threshold", prob_threshold, 0.5f);
  
  ncnn::set_cpu_powersave(powersave);
  static ncnn::VulkanDevice* g_vkdev = 0;
  static ncnn::VkAllocator* g_blob_vkallocator = 0;
  static ncnn::VkAllocator* g_staging_vkallocator = 0;
  if(enable_gpu)
  {
    ROS_INFO_STREAM(node_name << " with GPU_SUPPORT, selected gpu_device: " << gpu_device);
    g_vkdev = ncnn::get_gpu_device(gpu_device);
    g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
    g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    yolov3.opt.use_vulkan_compute = enable_gpu;
    yolov3.set_vulkan_device(g_vkdev);
  }
  else
  {
    ROS_WARN_STREAM(node_name << " running on CPU");
  }

  std::string models_path,model_file, param_file;
  nhLocal.param("models_path", models_path, std::string("/home/bingda/ncnn-assets/models/"));
  ROS_INFO("Assets path: %s", models_path.c_str());

  // original pretrained model from https://github.com/eric612/MobileNet-YOLO
  // param : https://drive.google.com/open?id=1V9oKHP6G6XvXZqhZbzNKL6FI_clRWdC-
  // bin : https://drive.google.com/open?id=1DBcuFCr-856z3FRQznWL_S5h-Aj3RawA
  // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
  nhLocal.param("model_file", model_file, std::string("mobilenetv2_yolov3.bin"));
  nhLocal.param("param_file", param_file, std::string("mobilenetv2_yolov3.param"));
  yolov3.load_param((models_path+param_file).c_str());
  yolov3.load_model((models_path+model_file).c_str());
  ROS_INFO("Loaded: %s", model_file.c_str());
  nhLocal.param("display_output", display_output, true);

  image_transport::ImageTransport it(n);
  image_pub = it.advertise("/ncnn_image", 1);
  obj_pub = n.advertise<object_information_msgs::Object>("/objects", 50);  
  image_transport::Subscriber video = it.subscribe("/usb_cam/image_raw", 1, imageCallback);
  
  if(enable_gpu)
  {
    ncnn::create_gpu_instance();
  }
  while (ros::ok()) {
    ros::spinOnce();
  }
  if(enable_gpu)
  {
    ncnn::destroy_gpu_instance();
  }  

  return 0;
}