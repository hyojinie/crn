#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		NormalizeLayerTest()
			: blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_data_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~NormalizeLayerTest() {
			delete blob_bottom_data_;
			delete blob_top_;
		}
		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

	TYPED_TEST(NormalizeLayerTest, TestForward) {
		// do nothing
	}

	TYPED_TEST(NormalizeLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		NormalizeLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-3); //, 1701
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}

}  // namespace caffe