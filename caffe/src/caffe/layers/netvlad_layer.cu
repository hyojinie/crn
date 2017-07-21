/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/

#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layers/netvlad_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	
template <typename Dtype>
__global__ void NetvladTranspose(const int n, const int d_local, const int d_area, Dtype* trans_mat, Dtype* orig_mat) {
		CUDA_KERNEL_LOOP(index, n) {
			const int p = index / d_local;
			const int ho = index % d_local;
			const int res_idx = ho * d_area + p;
			trans_mat[index] = orig_mat[res_idx];
			//const int label_value = static_cast<int>(label[n * spatial_dim + s]);
			//loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
		}
}


template <typename Dtype>
void NetvladLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	// Output (NetVlad)
	Dtype* top_data = top[0]->mutable_gpu_data();
	caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

	// Offsets (Cluster Center)
	const Dtype* clst_center = this->blobs_[0]->gpu_data();
	const Dtype* soft_assign = bottom[0]->gpu_data(); // *Note: cpu data
	const Dtype* bottom_data = bottom[1]->gpu_data();

	const int num = bottom[1]->num(); // batch size :shape(0)
	const int width = bottom[1]->width(); // batch size :shape(3)
	const int height = bottom[1]->height(); // batch size :shape(2)
	// int channels = bottom[1]->channels();
	const int area = width * height;

	const int dim = bottom[1]->count() / bottom[1]->num();  // width*height*channel (contains res of one image)

	NetvladParameter netvlad_param = this->layer_param_.netvlad_param();
	const int cluster_num = netvlad_param.cluster_num();      // number of clusters K
	const int local_dim = netvlad_param.local_dim();  // local feature dim D (e.g. 128 for SIFT), output of previous layer's depth
	const int vlad_dim = netvlad_param.vlad_dim(); // vlad dim, which should match D*K

	Dtype* residual_data = residual_.mutable_gpu_data();
	Dtype* tresidual_data = tresidual_.mutable_gpu_data();
	
	caffe_gpu_set(residual_.count(), Dtype(0), residual_data);
	caffe_gpu_set(tresidual_.count(), Dtype(0), tresidual_data);

	if (residual_.count() != dim || tresidual_.count() != dim){
		LOG(FATAL) << "temporal storage residual or tresidual does not match size of C*W*H (dim).";
	}

	for (int n = 0; n < num; ++n) {
		// for every image,
		for (int k = 0; k < cluster_num; ++k){
			// for every cluster,
			Dtype* k_cum_residue = top_data + n * vlad_dim + k * local_dim;
			caffe_gpu_memcpy(sizeof(Dtype)* dim, bottom_data + n * dim, residual_data);
			const Dtype* curr_center = clst_center + k * local_dim;

			// residual_data = feature - cluster center 
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, 1, (Dtype)1., curr_center, sum_multiplier_.gpu_data(), (Dtype)1., residual_data); //(512x(13x13))
			// assignment to this cluster 
			const Dtype* curr_assignment = soft_assign + n * (cluster_num * area) + k * area; // soft_assign: (100x64x13x13)

			/*
			NetvladTranspose<Dtype> << <CAFFE_GET_BLOCKS(mat_dim), CAFFE_CUDA_NUM_THREADS >> >(
				mat_dim, local_dim, area, tresidual_data, residual_data);
			CUDA_POST_KERNEL_CHECK;
			*/
			
			caffe_gpu_gemv(CblasNoTrans, local_dim, area, (Dtype)1., residual_data, curr_assignment, (Dtype)0., k_cum_residue);

		}
		
	}

}

template <typename Dtype>
void NetvladLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   
	//Backward_cpu(top, propagate_down, bottom);

	const Dtype* top_diff = top[0]->gpu_diff();
	// assign (a) : after softmax
	const Dtype* assign_data = bottom[0]->gpu_data();
	Dtype* assign_diff = bottom[0]->mutable_gpu_diff();
	// x data
	const Dtype* x_data = bottom[1]->gpu_data();
	Dtype* x_diff = bottom[1]->mutable_gpu_diff();
	// offset
	const Dtype* clst_center = this->blobs_[0]->gpu_data();
	Dtype* clst_center_diff = this->blobs_[0]->mutable_gpu_diff(); // offset diff

	const int num = bottom[1]->num(); // batch size :shape(0)
	const int width = bottom[1]->width(); // batch size :shape(3)
	const int height = bottom[1]->height(); // batch size :shape(2)
	// int channels = bottom[1]->channels();
	const int area = width * height;
	const int dim = bottom[1]->count() / bottom[1]->num();  // width*height*channel (contains res of one image)

	NetvladParameter netvlad_param = this->layer_param_.netvlad_param();
	const int cluster_num = netvlad_param.cluster_num();      // number of clusters K
	const int local_dim = netvlad_param.local_dim();  // local feature dim D (e.g. 128 for SIFT), output of previous layer's depth
	const int vlad_dim = netvlad_param.vlad_dim(); // vlad dim, which should match D*K

	Dtype* residual_data = residual_.mutable_gpu_data();
	caffe_gpu_set(residual_.count(), Dtype(0.0), residual_data); // for the safety

	caffe_gpu_set(vlad_dim, Dtype(0.0), clst_center_diff);

	Dtype sum_ak;

	for (int n = 0; n < num; ++n) {
		// for each batch
		const Dtype* x_mat = x_data + n* (local_dim * area);
		Dtype* x_diff_mat = x_diff + n* (local_dim * area);
		const Dtype* assign_mat = assign_data + n * (cluster_num * area);
		Dtype* assign_diff_mat = assign_diff + n * (cluster_num * area);
		const Dtype* top_diff_mat = top_diff + n * vlad_dim;

		// dz/da
		for (int k = 0; k < cluster_num; ++k){
			caffe_copy(dim, x_mat, residual_data); // (512x13x13)

			const Dtype* curr_center = clst_center + k * local_dim;
			const Dtype* top_diff_k = top_diff_mat + k * local_dim;  // Good. 
			Dtype* assign_diff_k = assign_diff_mat + k * area;
			const Dtype* assign_mat_k = assign_mat + k * area;

			// subtract cluster center
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, 1, (Dtype)1., curr_center, sum_multiplier_.gpu_data(), (Dtype)1., residual_data); //(512x(13x13))

			// sum wrt #local_dim
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, area, 1, local_dim, (Dtype)1., residual_data, top_diff_k, (Dtype)0., assign_diff_k); //((13x13)x1)
			
			// dz/dc
			Dtype* clst_center_diff_k = clst_center_diff + k * local_dim;
			
			caffe_gpu_dot<Dtype>(area, assign_mat_k, sum_multiplier_.gpu_data(), &sum_ak);

			caffe_gpu_axpy<Dtype>(local_dim, sum_ak, top_diff_k, clst_center_diff_k); // 1x 512

		}

		// dz/dx
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, cluster_num, (Dtype)1., top_diff_mat, assign_mat, (Dtype)0., x_diff_mat);
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(NetvladLayer);

}  // namespace caffe
