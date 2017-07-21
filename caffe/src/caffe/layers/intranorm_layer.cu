/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/intranorm_layer.hpp"

namespace caffe {

template <typename Dtype>
void IntranormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	IntranormParameter intranorm_param = this->layer_param_.intranorm_param();
	int cluster_num = intranorm_param.cluster_num();
	Dtype eps_ = Dtype(0.000000000001);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype normsqr;
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;

  int subdim = d / cluster_num;

  caffe_gpu_powx(n*d, bottom_data, Dtype(2), squared_data);
  for (int i=0; i<n; ++i) {
	  Dtype* curr_sqr_data = squared_data + i*d;
	  const Dtype* curr_bottom_data = bottom_data + i*d;
	  Dtype* curr_top_data = top_data + i*d;
	  for (int k = 0; k < cluster_num; ++k){
		  caffe_gpu_asum<Dtype>(subdim, curr_sqr_data + k*subdim, &normsqr);
		  caffe_gpu_scale<Dtype>(subdim, pow(normsqr + eps_, -0.5), curr_bottom_data + k*subdim, curr_top_data + k*subdim);
	  }
  }
}

template <typename Dtype>
void IntranormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	IntranormParameter intranorm_param = this->layer_param_.intranorm_param();
	int cluster_num = intranorm_param.cluster_num();

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->num();
  int d = top[0]->count() / n;

  int subdim = d / cluster_num;

  Dtype a;
  Dtype eps_ = Dtype(0.000000000001);
  for (int i=0; i<n; ++i) {
	  const Dtype* curr_top_data = top_data + i*d;
	  const Dtype* curr_top_diff = top_diff + i*d;
	  const Dtype* curr_bottom_data = bottom_data + i*d;
	  Dtype* curr_bottom_diff = bottom_diff + i*d;
	  for (int k = 0; k < cluster_num; ++k){
		  caffe_gpu_dot(subdim, curr_top_data + k*subdim, curr_top_diff + k*subdim, &a);
		  caffe_gpu_scale(subdim, a, curr_top_data + k*subdim, curr_bottom_diff + k*subdim);
		  caffe_gpu_sub(subdim, curr_top_diff + k*subdim, curr_bottom_diff + k*subdim, curr_bottom_diff + k*subdim);
		  caffe_gpu_dot(subdim, curr_bottom_data + k*subdim, curr_bottom_data + k*subdim, &a);
		  caffe_gpu_scale(subdim, Dtype(pow(a + eps_, -0.5)), curr_bottom_diff + k*subdim, curr_bottom_diff + k*subdim);
	  }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(IntranormLayer);


}  // namespace caffe
