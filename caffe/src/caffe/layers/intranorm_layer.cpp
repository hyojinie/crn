/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/
#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/intranorm_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void IntranormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		/*
	  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
	  bottom[0]->height(), bottom[0]->width());
	  squared_.Reshape(bottom[0]->num(), bottom[0]->channels(),
	  bottom[0]->height(), bottom[0]->width());
	  */
		top[0]->ReshapeLike(*bottom[0]);
		squared_.ReshapeLike(*bottom[0]);
		//LOG(INFO) << "NL Reshape Done.";
	}

	template <typename Dtype>
	void IntranormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		IntranormParameter intranorm_param = this->layer_param_.intranorm_param();
		int cluster_num = intranorm_param.cluster_num();

		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* squared_data = squared_.mutable_cpu_data();
		int n = bottom[0]->num();
		int d = bottom[0]->count() / n;

		int subdim = d / cluster_num;
		Dtype eps_ = Dtype(0.000000000001);
		caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
		for (int i = 0; i < n; ++i) {
			Dtype* curr_sqr_data = squared_data + i*d;
			const Dtype* curr_bottom_data = bottom_data + i*d;
			Dtype* curr_top_data = top_data + i*d;
			for (int k = 0; k < cluster_num; ++k){
				Dtype normsqr = caffe_cpu_asum<Dtype>(subdim, curr_sqr_data + k*subdim);
				caffe_cpu_scale<Dtype>(subdim, pow(normsqr + eps_, -0.5), curr_bottom_data + k*subdim, curr_top_data + k*subdim);
			}
		}
	}

	template <typename Dtype>
	void IntranormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		IntranormParameter intranorm_param = this->layer_param_.intranorm_param();
		int cluster_num = intranorm_param.cluster_num();

		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int n = top[0]->num();
		int d = top[0]->count() / n;

		int subdim = d / cluster_num;
		Dtype eps_ = Dtype(0.000000000001);
		for (int i = 0; i < n; ++i) {
			const Dtype* curr_top_data = top_data + i*d;
			const Dtype* curr_top_diff = top_diff + i*d;
			const Dtype* curr_bottom_data = bottom_data + i*d;
			Dtype* curr_bottom_diff = bottom_diff + i*d;
			for (int k = 0; k < cluster_num; ++k){
				Dtype a = caffe_cpu_dot(subdim, curr_top_data + k*subdim, curr_top_diff + k*subdim);
				caffe_cpu_scale(subdim, a, curr_top_data + k*subdim, curr_bottom_diff + k*subdim);
				caffe_sub(subdim, curr_top_diff + k*subdim, curr_bottom_diff + k*subdim, curr_bottom_diff + k*subdim);
				a = caffe_cpu_dot(subdim, curr_bottom_data + k*subdim, curr_bottom_data + k*subdim);
				caffe_cpu_scale(subdim, Dtype(pow(a + eps_, -0.5)), curr_bottom_diff + k*subdim, curr_bottom_diff + k*subdim);
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(IntranormLayer);
#endif

	INSTANTIATE_CLASS(IntranormLayer);
	REGISTER_LAYER_CLASS(Intranorm);

}  // namespace caffe
