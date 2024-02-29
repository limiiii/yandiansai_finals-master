// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
    std::vector<float> maskdata;
    cv::Mat mask;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
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

#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "cpu.h"
#include "gpu.h"
#include "object_information_msgs/Object.h"

ncnn::Net yolact;

object_information_msgs::Object objMsg;
ros::Publisher obj_pub;
image_transport::Publisher image_pub;
std::vector<Object> objects;
cv_bridge::CvImagePtr cv_ptr;
sensor_msgs::ImagePtr image_msg;
bool display_output;
bool enable_gpu;
int thread;

int target_size = 550;
float prob_threshold = 0.5f;
float nms_threshold = 0.45f;

static int detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects)
{
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolact.create_extractor();
    ex.set_num_threads(thread); //bd add this line
    ex.input("input.1", in);

    ncnn::Mat maskmaps;
    ncnn::Mat location;
    ncnn::Mat mask;
    ncnn::Mat confidence;

    ex.extract("619", maskmaps); // 138x138 x 32

    ex.extract("816", location);   // 4 x 19248
    ex.extract("818", mask);       // maskdim 32 x 19248
    ex.extract("820", confidence); // 81 x 19248

    int num_class = confidence.w;
    int num_priors = confidence.h;

    // make priorbox
    ncnn::Mat priorbox(4, num_priors);
    {
        const int conv_ws[5] = {69, 35, 18, 9, 5};
        const int conv_hs[5] = {69, 35, 18, 9, 5};

        const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
        const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

        float* pb = priorbox;

        for (int p = 0; p < 5; p++)
        {
            int conv_w = conv_ws[p];
            int conv_h = conv_hs[p];

            float scale = scales[p];

            for (int i = 0; i < conv_h; i++)
            {
                for (int j = 0; j < conv_w; j++)
                {
                    // +0.5 because priors are in center-size notation
                    float cx = (j + 0.5f) / conv_w;
                    float cy = (i + 0.5f) / conv_h;

                    for (int k = 0; k < 3; k++)
                    {
                        float ar = aspect_ratios[k];

                        ar = sqrt(ar);

                        float w = scale * ar / 550;
                        float h = scale / ar / 550;

                        // This is for backward compatibility with a bug where I made everything square by accident
                        // cfg.backbone.use_square_anchors:
                        h = w;

                        pb[0] = cx;
                        pb[1] = cy;
                        pb[2] = w;
                        pb[3] = h;

                        pb += 4;
                    }
                }
            }
        }
    }

    const float confidence_thresh = 0.05f;
    const float nms_threshold = 0.5f;
    const int keep_top_k = 200;

    std::vector<std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);

    for (int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence.row(i);
        const float* loc = location.row(i);
        const float* pb = priorbox.row(i);
        const float* maskdata = mask.row(i);

        // find class id with highest score
        // start from 1 to skip background
        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 0 || score <= confidence_thresh)
            continue;

        // CENTER_SIZE
        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float pb_cx = pb[0];
        float pb_cy = pb[1];
        float pb_w = pb[2];
        float pb_h = pb[3];

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * pb_w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * pb_h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, (float)(bgr.rows - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;
        obj.maskdata = std::vector<float>(maskdata, maskdata + mask.w);

        class_candidates[label].push_back(obj);
    }

    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    // keep_top_k
    if (keep_top_k < (int)objects.size())
    {
        objects.resize(keep_top_k);
    }

    // generate mask
    for (int i = 0; i < (int)objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask(maskmaps.h, maskmaps.w, CV_32FC1);
        {
            mask = cv::Scalar(0.f);

            for (int p = 0; p < maskmaps.c; p++)
            {
                const float* maskmap = maskmaps.channel(p);
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask.data;

                // mask += m * coeff
                for (int j = 0; j < maskmaps.w * maskmaps.h; j++)
                {
                    mp[j] += maskmap[j] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y = 0; y < img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x = 0; x < img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    return 0;
}
static const char* class_names[] = {"background",
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

static const unsigned char colors[81][3] = {
    {56, 0, 255},
    {226, 255, 0},
    {0, 94, 255},
    {0, 37, 255},
    {0, 255, 94},
    {255, 226, 0},
    {0, 18, 255},
    {255, 151, 0},
    {170, 0, 255},
    {0, 255, 56},
    {255, 0, 75},
    {0, 75, 255},
    {0, 255, 169},
    {255, 0, 207},
    {75, 255, 0},
    {207, 0, 255},
    {37, 0, 255},
    {0, 207, 255},
    {94, 0, 255},
    {0, 255, 113},
    {255, 18, 0},
    {255, 0, 56},
    {18, 0, 255},
    {0, 255, 226},
    {170, 255, 0},
    {255, 0, 245},
    {151, 255, 0},
    {132, 255, 0},
    {75, 0, 255},
    {151, 0, 255},
    {0, 151, 255},
    {132, 0, 255},
    {0, 255, 245},
    {255, 132, 0},
    {226, 0, 255},
    {255, 37, 0},
    {207, 255, 0},
    {0, 255, 207},
    {94, 255, 0},
    {0, 226, 255},
    {56, 255, 0},
    {255, 94, 0},
    {255, 113, 0},
    {0, 132, 255},
    {255, 0, 132},
    {255, 170, 0},
    {255, 0, 188},
    {113, 255, 0},
    {245, 0, 255},
    {113, 0, 255},
    {255, 188, 0},
    {0, 113, 255},
    {255, 0, 0},
    {0, 56, 255},
    {255, 0, 113},
    {0, 255, 188},
    {255, 0, 94},
    {255, 0, 18},
    {18, 255, 0},
    {0, 255, 132},
    {0, 188, 255},
    {0, 245, 255},
    {0, 169, 255},
    {37, 255, 0},
    {255, 0, 151},
    {188, 0, 255},
    {0, 255, 37},
    {0, 255, 0},
    {255, 0, 170},
    {255, 0, 37},
    {255, 75, 0},
    {0, 0, 255},
    {255, 207, 0},
    {255, 0, 226},
    {255, 245, 0},
    {188, 255, 0},
    {0, 255, 18},
    {0, 255, 75},
    {0, 255, 151},
    {255, 56, 0},
    {245, 255, 0}
};
static cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
  cv::Mat image = bgr.clone();

  int color_index = 0;

  for (size_t i = 0; i < objects.size(); i++)
  {
      const Object& obj = objects[i];

      if (obj.prob < prob_threshold)
          continue;

      const unsigned char* color = colors[color_index % 81];
      color_index++;

      cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

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

      // draw mask
      for (int y = 0; y < image.rows; y++)
      {
          const uchar* mp = obj.mask.ptr(y);
          uchar* p = image.ptr(y);
          for (int x = 0; x < image.cols; x++)
          {
              if (mp[x] == 255)
              {
                  p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                  p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                  p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
              }
              p += 3;
          }
      }
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
    detect_yolact(cv_ptr->image, objects);
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
      sprintf(text, "YOLACT FPS %.03f", fps);
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
  ros::init(argc, argv, "yolact_node"); /**/
  ros::NodeHandle nhLocal("~");  
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  int powersave;
  nhLocal.param("powersave", powersave, 0);  
  nhLocal.param("thread", thread, 2);     
  nhLocal.param("gpu_device", gpu_device, 0);
  nhLocal.param("enable_gpu", enable_gpu, false);
  nhLocal.param("target_size", target_size, 550);
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
    yolact.opt.use_vulkan_compute = enable_gpu;
    yolact.set_vulkan_device(g_vkdev);
  }
  else
  {
    ROS_WARN_STREAM(node_name << " running on CPU");
  }

  std::string models_path,model_file, param_file;
  nhLocal.param("models_path", models_path, std::string("/home/bingda/ncnn-assets/models/"));
  ROS_INFO("Assets path: %s", models_path.c_str());
  // original model converted from https://github.com/dbolya/yolact
  // yolact_resnet50_54_800000.pth
  // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
  nhLocal.param("model_file", model_file, std::string("yolact.bin"));
  nhLocal.param("param_file", param_file, std::string("yolact.param"));
  yolact.load_param((models_path+param_file).c_str());
  yolact.load_model((models_path+model_file).c_str());
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

