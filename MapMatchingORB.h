#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "matchers_zk.hpp"

#include "ros/ros.h"
#include <std_msgs/String.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "message_filters/subscriber.h"
#include "tf/transform_listener.h"
#include "tf/transform_broadcaster.h"
#include "tf/message_filter.h"
#include <stdio.h>

#include <string>
#include <functional>
#include <iostream>
#include <fstream>  
#include <set>
#include <math.h>

#define M_PI 3.14159265358979323846
using namespace std;
using namespace cv;
using namespace cv::detail;

std::string set_local_map_;
std::string set_static_map_;
std::string pose_update_topic_;

float initialpose_position_x=0;
float initialpose_position_y=0;
float initialpose_orientation_z=0;
float initialpose_orientation_w=0;

