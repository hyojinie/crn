/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/netvlad_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void NetvladLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	NetvladParameter netvlad_param = this->layer_param_.netvlad_param();

	vector<int> weight_shape(2);
	weight_shape[0] = netvlad_param.cluster_num();// 64
	weight_shape[1] = netvlad_param.local_dim();// 256

	this->blobs_.resize(1);
	// Initialize and fill the weights:
	// output channels x input channels per-group x kernel height x kernel width
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

	// optional weight filler
	shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(netvlad_param.weight_filler()));
	weight_filler->Fill(this->blobs_[0].get());

	// Propagate gradients to the parameters (as directed by backward pass).
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}


template <typename Dtype>
void NetvladLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	NetvladParameter netvlad_param = this->layer_param_.netvlad_param();

	int cluster_num = netvlad_param.cluster_num();
	int local_dim = netvlad_param.local_dim();
	int vlad_dim = netvlad_param.vlad_dim();

	if (vlad_dim != cluster_num * local_dim){
		LOG(FATAL) << "Netvlad: VLAD DIM do not match CLUSTER NUM times LOCAL FEATURE DIM.";
	}

	if (cluster_num != bottom[0]->channels()){
		LOG(FATAL) << "Netvlad: Bottom0 Channels (Softmax) do not match CLUSTER NUM.";
	}

	if (local_dim != bottom[1]->channels()){
		LOG(FATAL) << "Netvlad: Bottom1 Channels (Conv5) do not match LOCAL FEATURE DIM.";
	}

	if (bottom[0]->num() != bottom[1]->num()){
		LOG(FATAL) << "Netvlad: Batch size of two bottoms do not match.";
	}

	if (bottom[0]->width() != bottom[1]->width()){
		LOG(FATAL) << "Netvlad: Width of two bottoms do not match.";
	}

	if (bottom[0]->height() != bottom[1]->height()){
		LOG(FATAL) << "Netvlad: Height of two bottoms do not match.";
	}

	vector<int> top_dims(2);
	top_dims[0] = bottom[1]->num();
	top_dims[1] = vlad_dim;

	top[0]->Reshape(top_dims);
	
	vector<int> mult_dims(1, bottom[0]->count(2)); // width * height

	// LOG(INFO) << bottom[0]->count(2) << "=" << (bottom[0]->width() * bottom[0]->height());

	sum_multiplier_.Reshape(mult_dims);
	Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
	caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data); // 1 x (13*13)
	
	/*
	vector<int> identity_dims(2);
	identity_dims[0] = local_dim; // width * height
	identity_dims[1] = local_dim;
	identity_matrix_.Reshape(identity_dims);

	Dtype* identity_matrix = identity_matrix_.mutable_cpu_data();
	caffe_set(identity_matrix_.count(), Dtype(0.0), identity_matrix);

	for (int i = 0; i < local_dim; i++){
		identity_matrix[local_dim * i + i] = Dtype(1.0);
	}
	*/
	
	vector<int> resid_dims(1);
	resid_dims[0] = bottom[1]->count(1);
	/*
	resid_dims[0] = bottom[1]->channels();  
	resid_dims[1] = bottom[1]->height();
	resid_dims[2] = bottom[1]->width();
	*/
	
	vector<int> tresid_dims(1); 
	tresid_dims[0] = bottom[1]->count(1);
	/*
	tresid_dims[0] = bottom[1]->width();
	tresid_dims[1] = bottom[1]->height();
	tresid_dims[2] = bottom[1]->channels();
	*/
	
	//LOG(INFO) << "resid_dims[0] " << resid_dims[0];

	residual_.Reshape(resid_dims);
	tresidual_.Reshape(tresid_dims);

	Dtype* residual_data = residual_.mutable_cpu_data();
	Dtype* tresidual_data = tresidual_.mutable_cpu_data();

	caffe_set(residual_.count(), Dtype(0.0), residual_data); // for the safety
	caffe_set(tresidual_.count(), Dtype(0.0), tresidual_data); // for the safety
	//LOG(INFO) << "Reshape Done.";
}

template <typename Dtype>
void NetvladLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {


	// Output (NetVlad)
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top[0]->count(), Dtype(0.0), top_data);

	// Offsets (Cluster Center)
	const Dtype* clst_center = this->blobs_[0]->cpu_data();

	const Dtype* soft_assign = bottom[0]->cpu_data();
	const Dtype* bottom_data = bottom[1]->cpu_data();
	
	// int count = bottom[1]->count(); // total volume
	const int num = bottom[1]->num(); // batch size :shape(0)
	const int width = bottom[1]->width(); // batch size :shape(3)
	const int height = bottom[1]->height(); // batch size :shape(2)
	// int channels = bottom[1]->channels();
	const int area = width * height;

	//LOG(INFO) << area;
	
	// int num_local_p_batch = num*width*height;
	const int dim = bottom[1]->count() / bottom[1]->num();  // width*height*channel (contains res of one image)
	//LOG(INFO) << "dim" << dim;

	//const Dtype* identity_matrix = identity_matrix_.cpu_data();

	NetvladParameter netvlad_param = this->layer_param_.netvlad_param();
	const int cluster_num = netvlad_param.cluster_num();      // number of clusters K
	const int local_dim = netvlad_param.local_dim();  // local feature dim D (e.g. 128 for SIFT), output of previous layer's depth
	const int vlad_dim = netvlad_param.vlad_dim(); // vlad dim, which should match D*K

	//LOG(INFO) << "cluster_num" << cluster_num;
	//LOG(INFO) << "local_dim" << local_dim;
	//LOG(INFO) << "vlad_dim" << vlad_dim;

	// Blob<Dtype> residual(bottom[1]->shape());
	// Dtype* residual_data = residual.mutable_cpu_data();
	// caffe_copy(bottom[1]->count(), bottom_data, residual_data);

	Dtype* residual_data = residual_.mutable_cpu_data();
	
	Dtype* tresidual_data = tresidual_.mutable_cpu_data();

	caffe_set(residual_.count(), Dtype(0.0), residual_data); // for the safety
	caffe_set(tresidual_.count(), Dtype(0.0), tresidual_data); // for the safety

	if (residual_.count() != dim || tresidual_.count() != dim){
		LOG(FATAL) << "temporal storage residual or tresidual does not match size of C*W*H (dim).";
	}

	//LOG(INFO) << "init done.";

	for (int n = 0; n < num; ++n) {
		// for every image,
		for (int k = 0; k < cluster_num; ++k){
			// for every cluster,
			// LOG(INFO) << "st."; // hyo
			Dtype* k_cum_residue = top_data + n * vlad_dim + k * local_dim;  // Good. 

			// re-copy all local features (C*H*W) into residual_data (512x13x13)
			memcpy(residual_data, bottom_data + n * dim, sizeof(Dtype)* dim);

			const Dtype* curr_center = clst_center + k * local_dim; // Good.

			// residual_data = feature - cluster center 
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, 1, (Dtype)1., curr_center, sum_multiplier_.cpu_data(), (Dtype)1., residual_data); //(512x(13x13))
			
			// assignment to this cluster 
			const Dtype* curr_assignment = soft_assign + n * (cluster_num * area) + k * area; // soft_assign: (100x64x13x13)
			
			for (int p = 0; p < area; ++p){
				for (int ho = 0; ho < local_dim; ++ho){
					int tres_idx = p * local_dim + ho;
					int res_idx = ho * area + p;
					tresidual_data[tres_idx] = residual_data[res_idx];
				}
			}
			
			for (int p = 0; p < area; ++p){
				// for every position p in WxH, multipy assignments
				Dtype* local_residue = tresidual_data + local_dim * p; // 1 x 512
				caffe_cpu_axpby<Dtype>(local_dim, curr_assignment[p], local_residue, Dtype(1.0), k_cum_residue); // 1x 512
				// this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_); convolution
			}

			/* INTRA NORMALIZATION COMMENTED FOR DEBUGGING
			// intra-normalization to apply the same PCA by the authors
			Dtype eps_ = Dtype(0.0000001);
			Dtype intra_norm = sqrt(caffe_cpu_dot(local_dim, k_cum_residue, k_cum_residue));
			const Dtype a_ = Dtype(1.0) / (intra_norm + eps_);

			// caffe_scal(local_dim, a_, k_cum_residue); // this caused problem
			// caffe_cpu_axpby(local_dim, Dtype(0.0), k_cum_residue, a_, k_cum_residue);

			for (int k = 0; k < local_dim; k++)
			{
				k_cum_residue[k] = a_ * k_cum_residue[k];
			}
			*/

			/*Debug*/
			/*
			if (k == 0 && n == 0){
				LOG(INFO) << "residual_data[0]" << residual_data[0];
				LOG(INFO) << "residual_data[1]" << residual_data[1];
			}
			*/

		}

	}

}

template <typename Dtype>
void NetvladLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	// assign (a) : after softmax
	const Dtype* assign_data = bottom[0]->cpu_data();
	Dtype* assign_diff = bottom[0]->mutable_cpu_diff();
	// x data
	const Dtype* x_data = bottom[1]->cpu_data();
	Dtype* x_diff = bottom[1]->mutable_cpu_diff();
    // offset
	const Dtype* clst_center = this->blobs_[0]->cpu_data();
	Dtype* clst_center_diff = this->blobs_[0]->mutable_cpu_diff(); // offset diff
	
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

	Dtype* residual_data = residual_.mutable_cpu_data();
	caffe_set(residual_.count(), Dtype(0.0), residual_data); // for the safety

	for (int i = 0; i < vlad_dim; ++i)
		clst_center_diff[i] = 0;

	for (int n = 0; n < num; ++n) {
		// for each batch
		const Dtype* x_mat = x_data + n* (local_dim * area);
		Dtype* x_diff_mat = x_diff + n* (local_dim * area);
		const Dtype* assign_mat = assign_data + n * (cluster_num * area);
		Dtype* assign_diff_mat = assign_diff + n * (cluster_num * area);
		const Dtype* top_diff_mat = top_diff + n * vlad_dim;

		// dz/da
		for (int k = 0; k < cluster_num; ++k){

			memcpy(residual_data, x_mat, sizeof(Dtype)* dim); // (512x13x13)

			const Dtype* curr_center = clst_center + k * local_dim; 
			
			const Dtype* top_diff_k = top_diff_mat + k * local_dim;  // Good. 
			Dtype* assign_diff_k = assign_diff_mat + k * area;
			const Dtype* assign_mat_k = assign_mat + k * area;

			// subtract cluster center
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, 1, (Dtype)1., curr_center, sum_multiplier_.cpu_data(), (Dtype)1., residual_data); //(512x(13x13))
			// multiply top_diff
			
			//caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, area, 1, local_dim, (Dtype)1., residual_data, top_diff_k, (Dtype)0., assign_diff_k); //((13x13)x1)
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasTrans, area, 1, local_dim, (Dtype)1., residual_data, top_diff_k, (Dtype)0., assign_diff_k); //((13x13)x1)
			// sum wrt #local_dim

			// dz/dc
			Dtype* clst_center_diff_k = clst_center_diff + k * local_dim;

			Dtype sum_ak = (Dtype)0.;
			for (int a = 0; a < area; ++a){
				sum_ak += assign_mat_k[a];
			}

			for (int d = 0; d < local_dim; ++d){
				clst_center_diff_k[d] += sum_ak * top_diff_k[d]; // / num;
			}

			/*
			if (k == 0 && n == 0){
				LOG(INFO) << "residual_data[0]" << residual_data[0];
				LOG(INFO) << "residual_data[1]" << residual_data[1];
			}
			*/
		}

		// dz/dx
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, local_dim, area, cluster_num, (Dtype)1., top_diff_mat, assign_mat, (Dtype)0., x_diff_mat);

	}

	// top and bottom size mismatch!
	// Do nothing
	// memcpy(bottom_diff, top_diff, sizeof(Dtype)* (top[0]->count()));
	// caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
	//LOG(INFO) << "BW Done";
}


#ifdef CPU_ONLY
STUB_GPU(NetvladLayer);
#endif

INSTANTIATE_CLASS(NetvladLayer);
REGISTER_LAYER_CLASS(Netvlad);

}  // namespace caffe
