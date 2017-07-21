#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/grfilter_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GrfilterLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 GrfilterLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 13, 13)),
        blob_bottom_b_(new Blob<Dtype>(2, 9, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    blob_bottom_vec_.push_back(blob_bottom_a_);
    blob_bottom_vec_.push_back(blob_bottom_b_);
    blob_top_vec_.push_back(blob_top_);
  }
	 virtual ~GrfilterLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GrfilterLayerTest, TestDtypesAndDevices);

TYPED_TEST(GrfilterLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GrfilterParameter* grfilter_param = layer_param.mutable_grfilter_param();
  grfilter_param->set_grid_num(3);

  shared_ptr<GrfilterLayer<Dtype> > layer(
	  new GrfilterLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

/*
TYPED_TEST(GrfilterLayerTest, TestProd) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GrfilterParameter* grfilter_param = layer_param.mutable_grfilter_param();
  grfilter_param->set_grid_num(3);

  shared_ptr<GrfilterLayer<Dtype> > layer(
      new GrfilterLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  const int num = this->blob_top_->shape(0);
  const int channels = this->blob_top_->shape(1);
  const int height = this->blob_top_->shape(2);
  const int width = this->blob_top_->shape(3);
  const Dtype* in_data_a = this->blob_bottom_a_->cpu_data();
  const Dtype* in_data_b = this->blob_bottom_b_->cpu_data();
  for (int n = 0; n < num; ++n) {
	  for (int c = 0; c < channels; ++c) {
		  for (int i = 0; i < height*width; ++i) {
			  EXPECT_NEAR(data[i + height*width*c + n*channels*height*width], in_data_a[i + height*width*c + n*channels*height*width] * in_data_b[i + n*height*width], 1e-4);
		  }
	  }
  }
}
*/

TYPED_TEST(GrfilterLayerTest, TestStableProdGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  GrfilterParameter* grfilter_param = layer_param.mutable_grfilter_param();
  grfilter_param->set_grid_num(3);

  GrfilterLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
  //checker.CheckGradient(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
