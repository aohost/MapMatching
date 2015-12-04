#include "MapMatchingORB.h"

//***********************featurefind************************//
class FeaturesFinder_zk
{
public:
    virtual ~FeaturesFinder_zk() {}
    void operator ()(const Mat &image, ImageFeatures &features);
    void operator ()(const Mat &image, ImageFeatures &features, const std::vector<cv::Rect> &rois);
    virtual void collectGarbage() {}

protected:
    virtual void find(const Mat &image, ImageFeatures &features) = 0;
};

void FeaturesFinder_zk::operator ()(const Mat &image, ImageFeatures &features)
{
    find(image, features);
    features.img_size = image.size();
}


void FeaturesFinder_zk::operator ()(const Mat &image, ImageFeatures &features, const vector<Rect> &rois)
{
    vector<ImageFeatures> roi_features(rois.size());
    size_t total_kps_count = 0;
    int total_descriptors_height = 0;

    for (size_t i = 0; i < rois.size(); ++i)
    {
        find(image(rois[i]), roi_features[i]);
        total_kps_count += roi_features[i].keypoints.size();
        total_descriptors_height += roi_features[i].descriptors.rows;
    }

    features.img_size = image.size();
    features.keypoints.resize(total_kps_count);
    features.descriptors.create(total_descriptors_height,
                                roi_features[0].descriptors.cols,
                                roi_features[0].descriptors.type());

    int kp_idx = 0;
    int descr_offset = 0;
    for (size_t i = 0; i < rois.size(); ++i)
    {
        for (size_t j = 0; j < roi_features[i].keypoints.size(); ++j, ++kp_idx)
        {
            features.keypoints[kp_idx] = roi_features[i].keypoints[j];
            features.keypoints[kp_idx].pt.x += (float)rois[i].x;
            features.keypoints[kp_idx].pt.y += (float)rois[i].y;
        }
        Mat subdescr = features.descriptors.rowRange(
                descr_offset, descr_offset + roi_features[i].descriptors.rows);
        roi_features[i].descriptors.copyTo(subdescr);
        descr_offset += roi_features[i].descriptors.rows;
    }
}
//**********************orbfind***********************//
class OrbFeaturesFinder_zk : public FeaturesFinder_zk
{
public:
	OrbFeaturesFinder_zk(Size _grid_size = Size(1,1), int nfeatures=1500, float scaleFactor=1.3f, int nlevels=5); //comments by yufang: set nfeatures is the number of feature. the bigger number means more features.

private:
	void find(const Mat &image, ImageFeatures &features);

	Ptr<ORB> orb;
	Size grid_size;
};

OrbFeaturesFinder_zk::OrbFeaturesFinder_zk(Size _grid_size, int n_features, float scaleFactor, int nlevels)
{
    grid_size = _grid_size;
	//grid_size.area()= 1;
    orb = new ORB(n_features * (99 + grid_size.area())/100/grid_size.area(), scaleFactor, nlevels);//510

}

void OrbFeaturesFinder_zk::find(const Mat &image, ImageFeatures &features)
{
    Mat gray_image;

    CV_Assert((image.type() == CV_8UC3) || (image.type() == CV_8UC4) || (image.type() == CV_8UC1));

    if (image.type() == CV_8UC3) {
        cvtColor(image, gray_image, CV_BGR2GRAY);
    } else if (image.type() == CV_8UC4) {
        cvtColor(image, gray_image, CV_BGRA2GRAY);
    } else if (image.type() == CV_8UC1) {
        gray_image=image;
    } else {
        CV_Error(CV_StsUnsupportedFormat, "");
    }

    if (grid_size.area() == 1)
		   (*orb)(image, Mat(), features.keypoints, features.descriptors);
    else
    {
        features.keypoints.clear();
        features.descriptors.release();

        std::vector<KeyPoint> points;
        Mat descriptors;

        for (int r = 0; r < grid_size.height; ++r)
            for (int c = 0; c < grid_size.width; ++c)
            {
 				        if (c == 1)
		            {
		             continue;
                 }
                                 
                int xl = c * gray_image.cols / grid_size.width;
                int yl = r * gray_image.rows / grid_size.height;
                int xr = (c+1) * gray_image.cols / grid_size.width;
                int yr = (r+1) * gray_image.rows / grid_size.height;

                // LOGLN("OrbFeaturesFinder::find: gray_image.empty=" << (gray_image.empty()?"true":"false") << ", "
                //     << " gray_image.size()=(" << gray_image.size().width << "x" << gray_image.size().height << "), "
                //     << " yl=" << yl << ", yr=" << yr << ", "
                //     << " xl=" << xl << ", xr=" << xr << ", gray_image.data=" << ((size_t)gray_image.data) << ", "
                //     << "gray_image.dims=" << gray_image.dims << "\n");

                Mat gray_image_part=gray_image(Range(yl, yr), Range(xl, xr));
                // LOGLN("OrbFeaturesFinder::find: gray_image_part.empty=" << (gray_image_part.empty()?"true":"false") << ", "
                //     << " gray_image_part.size()=(" << gray_image_part.size().width << "x" << gray_image_part.size().height << "), "
                //     << " gray_image_part.dims=" << gray_image_part.dims << ", "
                //     << " gray_image_part.data=" << ((size_t)gray_image_part.data) << "\n");

                (*orb)(gray_image_part, Mat(), points, descriptors);

                features.keypoints.reserve(features.keypoints.size() + points.size());
                for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp)
                {
                    kp->pt.x += xl;
                    kp->pt.y += yl;
                    features.keypoints.push_back(*kp);
                }
                features.descriptors.push_back(descriptors);
            }
    }
}


//////////////////////////////////////////////////////

extern bool ControlFlag;

class BestOf2NearesMatcher_zk : public FeaturesMatcher
{
public:
    BestOf2NearesMatcher_zk(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 4,
                          int num_matches_thresh2 = 4);

    void collectGarbage();

protected:
    void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);
	
    int num_matches_thresh1_;
    int num_matches_thresh2_;
    Ptr<FeaturesMatcher> impl_;

};

typedef set<pair<int, int> > MatchesSet;


class CpuMatcher : public FeaturesMatcher
{
public:
	CpuMatcher(float match_conf) : FeaturesMatcher(true), match_conf_(match_conf) {}
	void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

private:
	float match_conf_;
};

void CpuMatcher::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{
	CV_Assert(features1.descriptors.type() == features2.descriptors.type());
	CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);

#ifdef HAVE_TEGRA_OPTIMIZATION
	if (tegra::match2nearest(features1, features2, matches_info, match_conf_))
     return;
#endif

	matches_info.matches.clear();

	//索引参数的结构，该构造函数所实例的快速搜索结构是根据参数params所指定的特定算法来构建的。
	//params是由IndexParams的派生类的引用。 KDTreeIndexParams，该方法对应的索引结构由一组随机kd树构成(randomized kd-trees)，它可以平行地进行搜索。
	Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
	Ptr<flann::SearchParams> searchParams = new flann::SearchParams();//查找

	if (features2.descriptors.depth() == CV_8U)
	{
		indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);//位置敏感哈希算法
		searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	}

	FlannBasedMatcher matcher(indexParams, searchParams);
	//BruteForceMatcher<HammingLUT> matcher;
	vector< vector<DMatch> > pair_matches;
	MatchesSet matches;
	//vector<DMatch> matches_zk;
	// Find 1->2 matches
	matcher.knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);//寻找每个查询特征关键点对应的k个最佳匹配
	//matcher.match(features1.descriptors ,features2.descriptors, matches_zk);


	///////////6.16 by zk
	Mat img_matcher;
	//Mat img_1 = imread("grayimage1.jpg");
	//Mat img_2 = imread("grayimage2.jpg");
	Mat img_1 = imread(set_local_map_);
	Mat img_2 = imread(set_static_map_);


	vector<pair<int, int> > intel;
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];

		////////////////////////////////////6.11 by zk 匹配坐标筛选
		Point2f p1 = features1.keypoints[m0.queryIdx].pt;
		Point2f p2 = features2.keypoints[m0.trainIdx].pt;

	}

	//--------------------------------------------------------------------------------
	// Find 1->2 matches
	matcher.knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
		{
			matches_info.matches.push_back(m0);
			matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
		}
	}

	// Find 2->1 matches
	pair_matches.clear();
	matcher.knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
		{
			if (matches.find(make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
			{
				matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
			}
		}
	}
	vector<DMatch> matches_tem = matches_info.matches;
	matches_info.matches.clear();
	bool flag = false;
	// double noe_time_start = getTickCount();
	for (int i = 0; i < matches_tem.size(); i++)
	{
		const DMatch& m0 = matches_tem[i];
		Point2f p1 = features1.keypoints[m0.queryIdx].pt;
		Point2f p2 = features2.keypoints[m0.trainIdx].pt;
		flag = false;

		for (int j = 0; j < matches_tem.size(); j++)
		{
			if (i == j)
			{
				continue;
			}
			const DMatch& m0_ = matches_tem[j];
			Point2f p1_ = features1.keypoints[m0_.queryIdx].pt;
			Point2f p2_ = features2.keypoints[m0_.trainIdx].pt;
			float distance_1 = sqrt(abs((p1.x - p1_.x)*(p1.x - p1_.x) + (p1.y - p1_.y)*(p1.y - p1_.y)));
			float distance_2 = sqrt(abs((p2.x - p2_.x)*(p2.x - p2_.x) + (p2.y - p2_.y)*(p2.y - p2_.y)));
			float ratio = distance_1 / distance_2;

			if (0.997 <= ratio &&ratio <= 1.003)
			{
				for (int k = 0; k < matches_tem.size(); k++)
				{
					if (k == j)
					{
						continue;
					}
					const DMatch& _m0 = matches_tem[k];
					Point2f _p1 = features1.keypoints[_m0.queryIdx].pt;
					Point2f _p2 = features2.keypoints[_m0.trainIdx].pt;
					float distance_1_ = sqrt(abs((p1_.x - _p1.x)*(p1_.x - _p1.x) + (p1_.y - _p1.y)*(p1_.y - _p1.y)));
					float distance_2_ = sqrt(abs((p2_.x - _p2.x)*(p2_.x - _p2.x) + (p2_.y - _p2.y)*(p2_.y - _p2.y)));
					float ratio_ = distance_1_ / distance_2_;
					if (0.995 <= ratio_ &&ratio_ <= 1.005)
					{
						pair<int, int> tem;
						tem.first = i;
						tem.second = j;
						intel.push_back(tem);
						matches_info.matches.push_back(m0);
						flag = true;
						break;
					}
				}


			}
			if (flag)
			{
				break;
			}
		}
	}
	
}


BestOf2NearesMatcher_zk::BestOf2NearesMatcher_zk(bool try_use_gpu, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{
	impl_ = new CpuMatcher(match_conf);
	is_thread_safe_ = impl_->isThreadSafe();
	num_matches_thresh1_ = num_matches_thresh1;
	num_matches_thresh2_ = num_matches_thresh2;
}


void BestOf2NearesMatcher_zk::match(const ImageFeatures &features1, const ImageFeatures &features2,
	MatchesInfo &matches_info)
{

	// Check if it makes sense to find homography

	if (!ControlFlag)
	{
		(*impl_)(features1, features2, matches_info);
		if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
			return;
		//Construct point-point correspondences for homography estimation
		Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
		Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);

		///////////////////////////////////////

		size_t j = 0;
		std::vector<DMatch>::iterator it = matches_info.matches.begin();

		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			const DMatch& m = matches_info.matches[i];

			Point2f p1 = features1.keypoints[m.queryIdx].pt;
			Point2f p2 = features2.keypoints[m.trainIdx].pt;


			p1.x -= features1.img_size.width * 0.5f;
			p1.y -= features1.img_size.height * 0.5f;
			src_points.at<Point2f>(0, static_cast<int>(j)) = p1;

			//p = features2.keypoints[m.trainIdx].pt;
			p2.x -= features2.img_size.width * 0.5f;
			p2.y -= features2.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, static_cast<int>(j)) = p2;

			j++;
		}

	
		matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, CV_RANSAC);


		// Find number of inliers
		matches_info.num_inliers = 0;
		for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
		if (matches_info.inliers_mask[i])
			matches_info.num_inliers++;



		// Check if we should try to refine motion
		if (matches_info.num_inliers < num_matches_thresh2_)
			return;

		// Construct point-point correspondences for inliers only
		src_points.create(1, matches_info.num_inliers, CV_32FC2);
		dst_points.create(1, matches_info.num_inliers, CV_32FC2);
		int inlier_idx = 0;

		//9.9 by zk
		vector<DMatch> neidian;
		///////////
		
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			if (!matches_info.inliers_mask[i])
				continue;

			const DMatch& m = matches_info.matches[i];
			//9.9by zk
			neidian.push_back(matches_info.matches[i]);
			////////////
			Point2f p = features1.keypoints[m.queryIdx].pt;
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<Point2f>(0, inlier_idx) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
			dst_points.at<Point2f>(0, inlier_idx) = p;

			inlier_idx++;
		}

		matches_info.H = findHomography(src_points, dst_points, CV_RANSAC);


		
	}

}

void BestOf2NearesMatcher_zk::collectGarbage()
{
	impl_->collectGarbage();
}


//////////////////////////////////////////
Mat pinjie(vector<ImageFeatures> features,vector<MatchesInfo> pairwise_matches)
{
	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);
	//estimate_demo(features, pairwise_matches, cameras); 

	Mat R;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		//Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	
		//cout<<"Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K()<<endl;
	}
	//cout<<"Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K()<<endl;
	return R;
}

/////////////////////////////////////////////////
void MyRotation(Mat& src, Mat& dst, float TransMat[3][3]);//pic ransform
void Coord_transform(double& x, double& y, double& angel, Mat H, Mat R);//

void poseupdateCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg);//added by Snow

#define cut_image_end   0
#define cut_image_start 0

bool ControlFlag = false;
bool try_gpu = false;
vector<string> img_names;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;

std::string save_graph_to;

int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.25f; //comments by yufang: the little the more key points;


int main(int argc, char** argv)
{
	ros::init(argc, argv, "map_matching_orb_node");
	ros::NodeHandle node_("~");

	node_.param("set_local_map", set_local_map_, std::string("/home/robot/ourland/hector_staticmap20151102/src/lidar_slam/map/local_20151111.pgm"));
	node_.param("set_static_map", set_static_map_, std::string("/home/robot/ourland/hector_staticmap20151102/lidar_slam/map/staticmap_20151111.pgm"));
	node_.param("pose_update_topic", pose_update_topic_, std::string("poseupdate"));
 
	//cout<<"local map: "<<set_local_map_<<endl;
	//cout<<"static map: "<<set_static_map_<<endl;

	//node_.param<std::string>("local_map", local_map_, std::string("/home/yufang/match_9_21/src/mapmatch/txt_local_map.pgm"));
	//node_.param<std::string>("static_map", static_map_, std::string("/home/yufang/match_9_21/src/mapmatch/txt_static_map.pgm"));
  	 
	//cout<<"local map: "<<local_map_<<endl;
	//cout<<"static map: "<<static_map_<<endl;

	ros::NodeHandle nh_;
	ros::Subscriber sub = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>(pose_update_topic_, 1, poseupdateCallback);
	ros::spin();
	
	//cv::waitKey(0);
	
	return 0;
}

void MyRotation(Mat& src, Mat& dst, float TransMat[3][3])
{
	CV_Assert(src.data);
	CV_Assert(src.depth() != sizeof(uchar));

	// calculate margin point of dst image  
	float left = 0;
	float right = 0;
	float top = 0;
	float down = 0;

	float x = src.cols * 1.0f;
	float y = 0.0f;
	float u1 = x * TransMat[0][0] + y * TransMat[0][1];
	float v1 = x * TransMat[1][0] + y * TransMat[1][1];
	x = src.cols * 1.0f;
	y = src.rows * 1.0f;
	float u2 = x * TransMat[0][0] + y * TransMat[0][1];
	float v2 = x * TransMat[1][0] + y * TransMat[1][1];
	x = 0.0f;
	y = src.rows * 1.0f;
	float u3 = x * TransMat[0][0] + y * TransMat[0][1];
	float v3 = x * TransMat[1][0] + y * TransMat[1][1];

	left = min(min(min(0.0f, u1), u2), u3);
	right = max(max(max(0.0f, u1), u2), u3);
	top = min(min(min(0.0f, v1), v2), v3);
	down = max(max(max(0.0f, v1), v2), v3);

	// create dst image  
	dst.create(int(abs(right - left)), int(abs(down - top)), src.type());


	CV_Assert(dst.channels() == src.channels());
	int channels = dst.channels();

	int i, j;
	uchar* p;
	uchar* q;
	for (i = 0; i < dst.rows; ++i)
	{
		p = dst.ptr<uchar>(i);
		for (j = 0; j < dst.cols; ++j)
		{
			// rotation  
			int x = (j + left)*TransMat[0][0] - (i + top)*TransMat[0][1]; // NOTE: adverse rotation here!!!  
			int y = -(j + left)*TransMat[1][0] + (i + top)*TransMat[1][1];


			if ((x >= 0) && (x < src.cols) && (y >= 0) && (y < src.rows))
			{
				q = src.ptr<uchar>(y);
				switch (channels)
				{
				case 1:
				{
						  p[j] = q[x];
						  break;
				}
				case 3:
				{
						  p[3 * j] = q[3 * x];
						  p[3 * j + 1] = q[3 * x + 1];
						  p[3 * j + 2] = q[3 * x + 2];
						  break;
				}
				}
			}
		}
	}
}

void Coord_transform(double& x, double& y, double& angle, Mat H, Mat R)
{
	Mat mat_src(1, 3, CV_64FC1);
	//Mat mat_src1(1, 1, CV_32FC1);
	mat_src.at<double>(0, 0) = x - 1024 * 0.5f;
	mat_src.at<double>(0, 0) = 0;
	mat_src.at<double>(0, 0) = 0;
	mat_src.at<double>(0, 1) = y - 1024 * 0.5f;
	mat_src.at<double>(0, 0) = 0;
	mat_src.at<double>(0, 0) = 0;
	mat_src.at<double>(0, 2) = 0;
	mat_src.at<double>(0, 0) = 0;
	mat_src.at<double>(0, 0) = 0;
	Mat mat_result;
	//Mat mat_src1(1, 3, CV_64FC1);
	/*cout << "channels:" << H.channels() << endl;
	cout << "type:" << H.type() << endl;
	cout << "depth:" << H.depth() << endl;
	cout << "elemSize:" << H.elemSize() << endl;
	cout << "elemSize1:" << H.elemSize1() << endl;*/
	Mat _H(1,3, CV_64FC1);
//	cvCopy(&H, &_H);
	//float *temp = _H.ptr<float>(0);
	//cout <<temp[0];
	//mat_src1.at<double>(0, 0) = mat_src.at<double>(0, 0)*H.at<double>(0, 0);
	//mat_src1.at<double>(0, 1) = mat_src.at<double>(0, 1)*H.at<double>(0, 1);
	//mat_src1.at<double>(0, 2) = mat_src.at<double>(0, 2)*H.at<double>(0, 2);
	//cout << H.at<float>(0, 0);
	double _x = (x - 1024 * 0.5f) * H.at<double>(0, 0) + H.at<double>(0,1)*(y-1024*0.5f)+ H.at<double>(0,2);
	double _y = (x - 1024 * 0.5f) * H.at<double>(1, 0) + H.at<double>(1, 1)*(y - 1024 * 0.5f) + H.at<double>(1, 2);
	//mat_src1.at<float>(0, 1) = mat_src.at<float>(0, 1)*H.at<float>(0, 1);
	//mat_src1.at<float>(0, 2) = mat_src.at<float>(0, 2)*H.at<float>(0, 2);

	x = _x+512;
	y = _y+512;



	float  result_z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
  result_z = result_z* 180 / 3.14159;
  angle = angle* 180 / 3.14159;
  
  angle = angle + result_z;
  if (angle > 180)
  {
   angle = angle - 2*180;
  }
  angle = angle*3.14159/180;
  
 	//added by Snow
  
  
	initialpose_position_x=(x-512)/20;
	initialpose_position_y=(512-y)/20;
  initialpose_orientation_z = sin(angle/2); //modified by yufang
  initialpose_orientation_w = cos(angle/2); //modified by yufang
 //
 

}

void poseupdateCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg)
{
	//???
	double app_start_time = getTickCount();
	cv::setBreakOnError(true);

	vector<Mat> all_imgs;

	img_names.push_back(set_local_map_);
	img_names.push_back(set_static_map_);
 
 std::cout<<set_local_map_;
 std::cout<<set_static_map_;

	// Check if have enough images
    int num_images = 2;

	double work_scale = 1, seam_scale = 1, compose_scale = 1;

	Ptr<FeaturesFinder_zk> finder;
	finder = new OrbFeaturesFinder_zk;
	//finder = new SurfFeaturesFinder();
	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Mat> gray_images(num_images);
	vector<Size> full_img_sizes(num_images);


	std::set<string> zk;
	std::set<string>::iterator it = zk.begin();
	for (int i = 0; i < num_images; ++i)
	{
		//full_img = all_imgs[i];
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		img = full_img;
		work_scale = 1;

		(*finder)(img, features[i]);

		features[i].img_idx = i;

		//size(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	vector<MatchesInfo> pairwise_matches;

	//	MatchesInfo temp_matches;
	BestOf2NearesMatcher_zk matcher(try_gpu, match_conf);

	//////////////5.12 by zk

	Mat mask_zk;
	const int num_images_zk = static_cast<int>(features.size()) - cut_image_end;

	CV_Assert(mask_zk.empty() || (mask_zk.type() == CV_8U && mask_zk.cols == num_images_zk && mask_zk.rows));
	Mat_<uchar> mask_(mask_zk);
	if (mask_.empty())
		mask_ = Mat::ones(num_images_zk, num_images_zk, CV_8U);

	bool zk_test = false;
	vector<pair<int, int> > near_pairs;
	/////////////////
	//vector< ImageFeatures >::iterator k = features.begin();

	//  features.erase(k); // ???????

	near_pairs.push_back(make_pair(0, 1));

	pairwise_matches.resize(num_images_zk * num_images_zk);
	MatchPairsBody_zk body(matcher, features, pairwise_matches, near_pairs, cut_image_end);

	body(Range(0, static_cast<int>(near_pairs.size())));
	matcher.collectGarbage();

	Mat R = pinjie(features, pairwise_matches);

	Mat pic_1 = imread(set_local_map_);

	Mat pic_2 = imread(set_static_map_);
 
  double x1 = 512 + msg->pose.pose.position.x*20; //updated by Snow
  double y1 = 512 - msg->pose.pose.position.y*20; //updated by Snow
  
	//double x2 = 200;
	//double y2 = 200;
	double angle = asin(msg->pose.pose.orientation.z)*2; //updated by Snow //modified by yufang
	Coord_transform(x1, y1, angle, pairwise_matches[1].H, R);

	float  result_z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));

	Mat dst;
	float alpha = (result_z-90)*3.14159 / 180;
	
	float transMat[3][3] = { { cos(alpha), -sin(alpha), 0 }, { sin(alpha), cos(alpha), 0 }, { 0, 0, 1 } };// !!! counter clockwise transform matrix, for clockwise rotation !!!  
	MyRotation(images[0], dst, transMat);

	//added by Snow
	ros::NodeHandle node_;
    ros::Publisher initialPosePublisher_;
    initialPosePublisher_ = node_.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1, false);
    geometry_msgs::PoseWithCovarianceStamped initial_pose_ = geometry_msgs::PoseWithCovarianceStamped();
    initial_pose_.header.frame_id = "map";
    
  for(;;)
  {
   (char)waitKey(1000);
    
   ros::Rate loop_rate(10);
   while (ros::ok())
    {
      cout << "< prepare to publish initial pose matched..." << endl; 
      initial_pose_.pose.pose.position.x = initialpose_position_x;
      initial_pose_.pose.pose.position.y = initialpose_position_y;
      initial_pose_.pose.pose.orientation.z = initialpose_orientation_z;
      initial_pose_.pose.pose.orientation.w = initialpose_orientation_w;
      ROS_INFO("initial pose with x: %f y: %f z: %f w: %f seq: %d ", initial_pose_.pose.pose.position.x, initial_pose_.pose.pose.position.y,initial_pose_.pose.pose.orientation.z,initial_pose_.pose.pose.orientation.w, initial_pose_.header.seq);
      initialPosePublisher_.publish(initial_pose_);
      cout << "< initial pose matched is published..." << endl;
      loop_rate.sleep();
     }
    break;
  }
}
