#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/intranorm_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class IntranormLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		IntranormLayerTest()
			: blob_bottom_data_(new Blob<Dtype>(10, 6, 1, 1)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);

			//caffe_set(3, Dtype(0), this->blob_bottom_data_->mutable_cpu_data());

			blob_bottom_vec_.push_back(blob_bottom_data_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~IntranormLayerTest() {
			delete blob_bottom_data_;
			delete blob_top_;
		}
		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(IntranormLayerTest, TestDtypesAndDevices);

	TYPED_TEST(IntranormLayerTest, TestForward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		IntranormParameter* intranorm_param = layer_param.mutable_intranorm_param();
		intranorm_param->set_cluster_num(2);
		IntranormLayer<Dtype> layer(layer_param);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		// test s
		for (int i = 0; i < this->blob_bottom_data_->num(); ++i) {
			for (int k = 0; k < 2; ++k) {
				Dtype sum = 0;
				for (int j = 0; j < 3; ++j){
					Dtype elem = this->blob_top_->data_at(i, k * 3 + j, 0, 0);
					sum += (elem * elem);
				}
				sum = sqrt(sum);
				Dtype e1 = this->blob_top_->data_at(i, k * 3 + 0, 0, 0);
				Dtype e2 = this->blob_top_->data_at(i, k * 3 + 1, 0, 0);
				Dtype e3 = this->blob_top_->data_at(i, k * 3 + 2, 0, 0);
				Dtype b1 = this->blob_bottom_data_->data_at(i, k * 3 + 0, 0, 0);
				Dtype b2 = this->blob_bottom_data_->data_at(i, k * 3 + 1, 0, 0);
				Dtype b3 = this->blob_bottom_data_->data_at(i, k * 3 + 2, 0, 0);
				EXPECT_GE(sum, 0.999) << "debug: (" << i <<" ," << k <<")" << e1 << " " << e2 << " " << e3;
				EXPECT_LE(sum, 1.001) << "debug: (" << i <<" , " << k << ")" << e1 << " " << e2 << " " << e3;
				EXPECT_GE(sum, 0.999) << "debug: (" << i << " ," << k << ")" << b1 << " " << b2 << " " << b3;
				EXPECT_LE(sum, 1.001) << "debug: (" << i << " , " << k << ")" << b1 << " " << b2 << " " << b3;
			}
		}
	}

	TYPED_TEST(IntranormLayerTest, TestGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;

		IntranormParameter* intranorm_param = layer_param.mutable_intranorm_param();
		intranorm_param->set_cluster_num(2);

		IntranormLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-3); //, 1701
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}

}  // namespace caffe