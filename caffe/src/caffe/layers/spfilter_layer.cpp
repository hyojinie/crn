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
	void SpfilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(top.size(), bottom.size() - 1);
	}

	template <typename Dtype>
	void SpfilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// bottom[0...k-1] are the blobs to filter
		// bottom[last] is the "selector_blob"
		CHECK_EQ(top.size(), bottom.size() - 1);

		int selector_index = bottom.size() - 1;
		for (int i = 0; i < bottom.size() - 1; ++i) {
			CHECK_EQ(bottom[selector_index]->shape(0), bottom[i]->shape(0)) <<
				"Each bottom should have the same batch size as the selector blob";
			CHECK_EQ((bottom[selector_index]->count() / bottom[selector_index]->shape(0)), (bottom[i]->shape(2))*(bottom[i]->shape(3))) <<
				"Each bottom should have the same spatial dimension as the selector blob";
		}
		for (int t = 0; t < top.size(); ++t) {
			top[t]->ReshapeLike(*bottom[t]);
		}
		temp_.ReshapeLike(*bottom[0]); // all bottom should have same shape NxCxHxW
		Dtype* temp_data = temp_.mutable_cpu_data();
		caffe_set(temp_.count(), Dtype(0), temp_data);
	}

	template <typename Dtype>
	void SpfilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int selector_index = bottom.size() - 1;
		const Dtype* select_data = bottom[selector_index]->cpu_data();
		// forward all filtered items for all bottoms but the Selector (bottom[last])
		for (int t = 0; t < top.size(); ++t) {
			const Dtype* bottom_data = bottom[t]->cpu_data();
			Dtype* top_data = top[t]->mutable_cpu_data();

			const int num = bottom[t]->shape(0);
			const int channels = bottom[t]->shape(1);
			const int dim = bottom[t]->count() / num;
			const int area = (bottom[t]->shape(2))*(bottom[t]->shape(3));

			for (int n = 0; n < num; ++n) {
				const Dtype* curr_select_data = select_data + n*area;
				const Dtype* curr_bottom_data = bottom_data + n*dim;
				Dtype* curr_top_data = top_data + n*dim;

				for (int c = 0; c < channels; ++c) {
					caffe_mul(area, curr_select_data, curr_bottom_data + c*area, curr_top_data + c*area);
				}
			}
		}
	}

	template <typename Dtype>
	void SpfilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const int selector_index = bottom.size() - 1;
		const Dtype* select_data = bottom[selector_index]->cpu_data();
		Dtype* select_diff = bottom[selector_index]->mutable_cpu_diff();

		Dtype* temp_data = temp_.mutable_cpu_data();

		for (int i = 0; i < bottom[selector_index]->count(); ++i)
			select_diff[i] = 0;

		for (int t = 0; t < top.size(); t++) {
			const Dtype* top_data = top[t]->cpu_data();
			const Dtype* bottom_data = bottom[t]->cpu_data();
			Dtype* top_diff = top[t]->mutable_cpu_diff();
			Dtype* bottom_diff = bottom[t]->mutable_cpu_diff();

			const int count = bottom[t]->count();
			const int num = bottom[t]->shape(0);
			const int channels = bottom[t]->shape(1);
			const int dim = count / num;
			const int area = (bottom[t]->shape(2))*(bottom[t]->shape(3));

			// if (propagate_down[t])

			// diff += top_data.*top_diff(t)
			caffe_mul(count, bottom_data, top_diff, temp_data);

			for (int n = 0; n < num; n++) {
				const Dtype* curr_select_data = select_data + n*area;
				Dtype* curr_select_diff = select_diff + n*area;

				const Dtype* curr_bottom_data = bottom_data + n*dim;
				Dtype* curr_bottom_diff = bottom_diff + n*dim;
				Dtype* curr_top_diff = top_diff + n*dim;

				for (int c = 0; c < channels; ++c) {
					// for non-selector
					caffe_copy(area, curr_select_data, curr_bottom_diff + c*area);
					// for selector
					caffe_axpy(area, Dtype(1), temp_data + n*dim + c*area, curr_select_diff);
				}
			}
			// for non-selector
			caffe_mul(count, bottom_diff, top_diff, bottom_diff);

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SpfilterLayer);
#endif

	INSTANTIATE_CLASS(SpfilterLayer);
	REGISTER_LAYER_CLASS(Spfilter);

}  // namespace caffe
