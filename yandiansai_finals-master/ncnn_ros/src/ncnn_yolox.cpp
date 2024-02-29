// This file is wirtten base on the following file:
// https://github.com/Tencent/ncnn/blob/master/examples/yolov5.cpp
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
// ------------------------------------------------------------------------------
// Copyright (C) 2020-2021, Megvii Inc. All rights reserved.

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

#define YOLOX_NMS_THRESH  0.45 // nms threshold
#define YOLOX_CONF_THRESH 0.25 // threshold of bounding box prob
#define YOLOX_TARGET_SIZE 640  // target image size after resize, might use 416 for small model

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;
    const int num_class = feat_blob.w - 5;
    const int num_anchors = grid_strides.size();

    const float* feat_ptr = feat_blob.channel(0);
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        feat_ptr += feat_blob.w;

    } // point anchor loop
}

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "cpu.h"
#include "gpu.h"
#include "object_information_msgs/Object.h"

ncnn::Net yolox;

object_information_msgs::Object objMsg;
ros::Publisher obj_pub;
image_transport::Publisher image_pub;
std::vector<Object> objects;
cv_bridge::CvImagePtr cv_ptr;
sensor_msgs::ImagePtr image_msg;
bool display_output;
bool enable_gpu;
int thread;

int target_size = 640;
float prob_threshold = 0.5f;
float nms_threshold = 0.45f;


static int detect_yolox(const cv::Mat& bgr, std::vector<Object>& objects)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    // different from yolov5, yolox only pad on bottom and right side,
    // which means users don't need to extra padding info to decode boxes coordinate.
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    ncnn::Extractor ex = yolox.create_extractor();
    ex.set_num_threads(thread); //bd add this line
    ex.input("images", in_pad);

    std::vector<Object> proposals;

    {
        ncnn::Mat out;
        ex.extract("output", out);

        static const int stride_arr[] = {8, 16, 32}; // might have stride=64 in YOLOX
        std::vector<int> strides(stride_arr, stride_arr + sizeof(stride_arr) / sizeof(stride_arr[0]));
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_yolox_proposals(grid_strides, out, prob_threshold, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
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
    detect_yolox(cv_ptr->image, objects);
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
      sprintf(text, "YOLO X FPS %.03f", fps);
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
  ros::init(argc, argv, "yolox_node"); /**/
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  int powersave;
  nhLocal.param("powersave", powersave, 0);  
  nhLocal.param("thread", thread, 2);   
  nhLocal.param("gpu_device", gpu_device, 0);
  nhLocal.param("enable_gpu", enable_gpu, false);
  nhLocal.param("target_size", target_size, 640);
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
    yolox.opt.use_vulkan_compute = enable_gpu;
    yolox.set_vulkan_device(g_vkdev);
  }
  else
  {
    ROS_WARN_STREAM(node_name << " running on CPU");
  }

  std::string models_path,model_file, param_file;
  nhLocal.param("models_path", models_path, std::string("/home/bingda/ncnn-assets/models/"));
  ROS_INFO("Assets path: %s", models_path.c_str());
  // original pretrained model from https://github.com/Megvii-BaseDetection/YOLOX
  // ncnn model param: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s_ncnn.tar.gz
  // NOTE that newest version YOLOX remove normalization of model (minus mean and then div by std),
  // which might cause your model outputs becoming a total mess, plz check carefully.
  nhLocal.param("model_file", model_file, std::string("yolox.bin"));
  nhLocal.param("param_file", param_file, std::string("yolox.param"));
  yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
  yolox.load_param((models_path+param_file).c_str());
  yolox.load_model((models_path+model_file).c_str());
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

