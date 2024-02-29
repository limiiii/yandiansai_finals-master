cmake_minimum_required(VERSION 3.0.2)
project(ncnn_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

set(ncnn_DIR "~/ncnn/build/install/lib/cmake/ncnn/")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  image_transport
  image_geometry
  sensor_msgs
  message_generation
  object_information_msgs
)


catkin_package(
  CATKIN_DEPENDS
  roscpp
  roslib
  cv_bridge
  image_transport
  image_geometry
  sensor_msgs
  object_information_msgs
)

include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
)

find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

# NanoDet
add_executable(nanodet_node src/ncnn_nanodet.cpp)
target_link_libraries(nanodet_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLO v3
add_executable(yolov3_node src/ncnn_yolov3.cpp)
target_link_libraries(yolov3_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLO v5
add_executable(yolov5_node src/ncnn_yolov5.cpp)
target_link_libraries(yolov5_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLO X
add_executable(yolox_node src/ncnn_yolox.cpp)
target_link_libraries(yolox_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLO v7
add_executable(yolov7_node src/ncnn_yolov7.cpp)
target_link_libraries(yolov7_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLACT
add_executable(yolact_node src/ncnn_yolact.cpp)
add_dependencies( yolact_node object_information_msgs_generate_messages_cpp)
target_link_libraries(yolact_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})