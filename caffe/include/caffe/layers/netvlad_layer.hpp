/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/
#ifndef CAFFE_NETVLAD_LAYER_HPP_
#define CAFFE_NETVLAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class NetvladLayer : public Layer<Dtype> {
	public:
		explicit NetvladLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Netvlad"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//virtual inline bool reverse_dimensions() { return false; }
		// virtual void compute_output_shape();

		Blob<Dtype> identity_matrix_;
		/// sum_multiplier is used to carry out sum using BLAS
		Blob<Dtype> sum_multiplier_;

		Blob<Dtype> residual_;
		Blob<Dtype> tresidual_;
	};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
