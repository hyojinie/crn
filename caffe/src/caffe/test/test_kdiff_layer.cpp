#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kdiff_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class KdiffLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		KdiffLayerTest()
			: blob_bottom_data_(new Blob<Dtype>(16, 1, 1, 5)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~KdiffLayerTest() {
			delete blob_bottom_data_;
			delete blob_top_;
		}
		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(KdiffLayerTest, TestDtypesAndDevices);

	TYPED_TEST(KdiffLayerTest, TestForward) {
		// do nothing
	}

	TYPED_TEST(KdiffLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;

		LayerParameter layer_param;
		
		KdiffParameter* kdiff_param = layer_param.mutable_kdiff_param();
		kdiff_param->set_cluster_num(3);
		kdiff_param->mutable_weight_filler()->set_type("gaussian");
		
		KdiffLayer<Dtype> layer(layer_param);
		
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		
		GradientChecker<Dtype> checker(1e-2, 1e-2); //, 1701
		checker.CheckGradient(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
	}

}  // namespace caffe