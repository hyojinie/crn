/* UNC Software “Learned Contextual Feature Reweighting for Image Geo-Localization”
Copyright (C) 2017 The University of North Carolina at Chapel Hill
All rights reserved.
Written by Hyo Jin Kim (hyojin@cs.unc.edu)
*/
#include <vector>

#include "caffe/layers/spfilter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpfilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//Forward_cpu(bottom, top);
	
	const int selector_index = bottom.size() - 1;
	const Dtype* select_data = bottom[selector_index]->gpu_data();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    const Dtype* bottom_data = bottom[t]->gpu_data();
    Dtype* top_data = top[t]->mutable_gpu_data();

	const int num = bottom[t]->shape(0);
	const int channels = bottom[t]->shape(1);
	const int dim = bottom[t]->count() / num;
	const int area = (bottom[t]->shape(2))*(bottom[t]->shape(3));

    for (int n = 0; n < num; ++n) {
		const Dtype* curr_select_data = select_data + n*area;
		const Dtype* curr_bottom_data = bottom_data + n*dim;
		Dtype* curr_top_data = top_data + n*dim;

		for (int c = 0; c < channels; ++c) {
			caffe_gpu_mul(area, curr_select_data, curr_bottom_data + c*area, curr_top_data + c*area);
		}
    }
  }
  
}

template <typename Dtype>
void SpfilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	//Backward_cpu(top, propagate_down, bottom);
	
	const int selector_index = bottom.size() - 1;
	const Dtype* select_data = bottom[selector_index]->gpu_data();
	Dtype* select_diff = bottom[selector_index]->mutable_gpu_diff();

	Dtype* temp_data = temp_.mutable_gpu_data();

	caffe_gpu_set(bottom[selector_index]->count(), Dtype(0.0), select_diff);

  for (int t = 0; t < top.size(); t++) {
	  const Dtype* top_data = top[t]->gpu_data();
	  const Dtype* bottom_data = bottom[t]->gpu_data();
	  Dtype* top_diff = top[t]->mutable_gpu_diff();
	  Dtype* bottom_diff = bottom[t]->mutable_gpu_diff();

	  const int count = bottom[t]->count();
	  const int num = bottom[t]->shape(0);
	  const int channels = bottom[t]->shape(1);
	  const int dim = count / num;
	  const int area = (bottom[t]->shape(2))*(bottom[t]->shape(3));

	  // if (propagate_down[t])
	  caffe_gpu_mul(count, bottom_data, top_diff, temp_data);

	  for (int n = 0; n < num; n++) {
		  const Dtype* curr_select_data = select_data + n*area;
		  Dtype* curr_select_diff = select_diff + n*area;

		  Dtype* curr_bottom_diff = bottom_diff + n*dim;

		  for (int c = 0; c < channels; ++c) {
			  // for non-selector
			  caffe_copy(area, curr_select_data, curr_bottom_diff + c*area);
			  // for selector
			  caffe_gpu_axpy(area, Dtype(1), temp_data + n*dim + c*area, curr_select_diff);
		  }
	  }

	  // for non-selector
	  caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);

    }
	
  }


INSTANTIATE_LAYER_GPU_FUNCS(SpfilterLayer);

}  // namespace caffe
