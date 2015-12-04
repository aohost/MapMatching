#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>

#include <opencv2/opencv_modules.hpp>
#include "opencv2/core/core.hpp"   
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"	


using namespace std;
using namespace cv;
using namespace detail;

extern bool ControlFlag;

struct MatchPairsBody_zk : ParallelLoopBody
{
    MatchPairsBody_zk(FeaturesMatcher &_matcher, vector<ImageFeatures> &_features,
                   vector<MatchesInfo> &_pairwise_matches, vector<pair<int,int> > &_near_pairs
				   ,int _cut_image_end)
            : matcher(_matcher), features(_features),
			pairwise_matches(_pairwise_matches), near_pairs(_near_pairs), cut_image_end(_cut_image_end) {}

	
    void operator ()(const Range &r) const
    {

		vector<int> bad_idx;
		vector<int> bad_from;
		vector<int> bad_to;

        const int num_images = static_cast<int>(features.size())-cut_image_end;
        for (int i = r.start; i < r.end; ++i)
        {
			int from = near_pairs[i].first;
			int to = near_pairs[i].second;
			int pair_idx = from*num_images + to;
			//int pair_idx = from;

			matcher(features[from], features[to], pairwise_matches[pair_idx]);
			pairwise_matches[pair_idx].src_img_idx = from;
			pairwise_matches[pair_idx].dst_img_idx = to;

			size_t dual_pair_idx = to*num_images + from;

			pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
			pairwise_matches[dual_pair_idx].src_img_idx = to;
			pairwise_matches[dual_pair_idx].dst_img_idx = from;

			if (!pairwise_matches[pair_idx].H.empty())
				pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

			for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
				std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
				pairwise_matches[dual_pair_idx].matches[j].trainIdx);

        }
		
    }

    FeaturesMatcher &matcher;
    //const vector<ImageFeatures> &features;
	vector<ImageFeatures> &features;
    vector<MatchesInfo> &pairwise_matches;
    vector<pair<int,int> > &near_pairs;
	int cut_image_end;

private:
    void operator =(const MatchPairsBody_zk&);
};


