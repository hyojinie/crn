#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define pi 3.14159265358979323846

namespace caffe {

	double deg2rad(double deg) {
		return (deg * pi / 180);
	}

	template <typename Dtype>
	ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int new_height = this->layer_param_.image_data_param().new_height();
		const int new_width = this->layer_param_.image_data_param().new_width();
		const bool is_color = this->layer_param_.image_data_param().is_color();
		string root_folder = this->layer_param_.image_data_param().root_folder();

		// hyo
		const bool use_gps = this->layer_param_.image_data_param().use_gps();
		if (use_gps){
			LOG(INFO) << "Using GPS tags for finding hard negatives.";
		}

		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";
		// Read the file with filenames and labels
		const string& source = this->layer_param_.image_data_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		string filename;

		cv::Mat cv_img;
		// hyo
		if (use_gps){
			double gps_lat;
			double gps_lon;
			while (infile >> filename >> gps_lat >> gps_lon) {
				tlines_.push_back(make_triplet(filename, deg2rad(gps_lat), deg2rad(gps_lon)));
			}
			if (this->layer_param_.image_data_param().shuffle()) {
				// randomly shuffle data
				LOG(INFO) << "Shuffling data";
				const unsigned int prefetch_rng_seed = caffe_rng_rand();
				prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
				ShuffleImagesTri();
			}
			LOG(INFO) << "A total of " << tlines_.size() << " images.";

			tlines_id_ = 0;
			// Check if we would need to randomly skip a few data points
			if (this->layer_param_.image_data_param().rand_skip()) {
				unsigned int skip = caffe_rng_rand() %
					this->layer_param_.image_data_param().rand_skip();
				LOG(INFO) << "Skipping first " << skip << " data points.";
				CHECK_GT(tlines_.size(), skip) << "Not enough points to skip";
				tlines_id_ = skip;
			}
			// Read an image, and use it to initialize the top blob.
			cv_img = ReadImageToCVMat(root_folder + tlines_[tlines_id_].first,
				new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << tlines_[tlines_id_].first;
		}
		else{
			int label;
			while (infile >> filename >> label) {
				lines_.push_back(std::make_pair(filename, label));
			}
			if (this->layer_param_.image_data_param().shuffle()) {
				// randomly shuffle data
				LOG(INFO) << "Shuffling data";
				const unsigned int prefetch_rng_seed = caffe_rng_rand();
				prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
				ShuffleImages();
			}
			LOG(INFO) << "A total of " << lines_.size() << " images.";

			lines_id_ = 0;
			// Check if we would need to randomly skip a few data points
			if (this->layer_param_.image_data_param().rand_skip()) {
				unsigned int skip = caffe_rng_rand() %
					this->layer_param_.image_data_param().rand_skip();
				LOG(INFO) << "Skipping first " << skip << " data points.";
				CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
				lines_id_ = skip;
			}
			// Read an image, and use it to initialize the top blob.
			cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		}

		// Use data_transformer to infer the expected blob shape from a cv_image.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.image_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		top_shape[0] = batch_size;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		top[0]->Reshape(top_shape);

		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();

		if (use_gps){
			// gps
			vector<int> gps_lat_shape(1, batch_size); // one int with batch size as its value
			top[1]->Reshape(gps_lat_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].gps_lat_.Reshape(gps_lat_shape);
			}
			vector<int> gps_lon_shape(1, batch_size);
			top[2]->Reshape(gps_lon_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].gps_lon_.Reshape(gps_lon_shape);
			}
		}
		else {
			// label
			vector<int> label_shape(1, batch_size);
			top[1]->Reshape(label_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].label_.Reshape(label_shape);
			}
		}
	}

	// hyo
	template <typename Dtype>
	int ImageDataLayer<Dtype>::myrandomdata(int i) { return caffe_rng_rand() % i; }

	template <typename Dtype>
	void ImageDataLayer<Dtype>::ShuffleImages() {
		
		// hyo
		int pairsize = this->layer_param_.image_data_param().pair_size();
		
		if (pairsize == 1){
			caffe::rng_t* prefetch_rng =
				static_cast<caffe::rng_t*>(prefetch_rng_->generator());
			shuffle(lines_.begin(), lines_.end(), prefetch_rng);
		}
		else{
			const int num_images = lines_.size();
			DLOG(INFO) << "My Shuffle.";
			vector<std::pair<std::string, int> > plines_;
			vector<int> pnum;

			for (int i = 0; i < num_images / pairsize; i++)
			{
				pnum.push_back(i);
			}
			std::random_shuffle(pnum.begin(), pnum.end(), ImageDataLayer<Dtype>::myrandomdata);
			plines_.clear();
			for (int i = 0; i < num_images / pairsize; i++)
			{
				for (int j = 0; j < pairsize; j++)
				{
					plines_.push_back(lines_[pnum[i] * pairsize + j]);
				}
			}
			lines_ = plines_;
		}
	}

	template <typename Dtype>
	void ImageDataLayer<Dtype>::ShuffleImagesTri() {
		int pairsize = this->layer_param_.image_data_param().pair_size();

		if (pairsize == 1){
			caffe::rng_t* prefetch_rng =
				static_cast<caffe::rng_t*>(prefetch_rng_->generator());
			shuffle(tlines_.begin(), tlines_.end(), prefetch_rng);
		}
		else
		{
			const int num_images = tlines_.size();
			DLOG(INFO) << "My Shuffle.";
			vector<Triplet> ptlines_;
			vector<int> ptnum;

			for (int i = 0; i < num_images / pairsize; i++)
			{
				ptnum.push_back(i);
			}
			std::random_shuffle(ptnum.begin(), ptnum.end(), ImageDataLayer<Dtype>::myrandomdata);
			ptlines_.clear();
			for (int i = 0; i < num_images / pairsize; i++)
			{
				for (int j = 0; j < pairsize; j++)
				{
					ptlines_.push_back(tlines_[ptnum[i] * pairsize + j]);
				}
			}
			tlines_ = ptlines_;
		}
		
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		ImageDataParameter image_data_param = this->layer_param_.image_data_param();
		const int batch_size = image_data_param.batch_size();
		const int new_height = image_data_param.new_height();
		const int new_width = image_data_param.new_width();
		const bool is_color = image_data_param.is_color();
		string root_folder = image_data_param.root_folder();

		// hyo
		const bool use_gps = this->layer_param_.image_data_param().use_gps();
		cv::Mat cv_img;
		const int tlines_size = tlines_.size();
		const int lines_size = lines_.size();
		if (use_gps){
			// Reshape according to the first image of each batch
			// on single input batches allows for inputs of varying dimension.
			cv_img = ReadImageToCVMat(root_folder + tlines_[tlines_id_].first,
				new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << tlines_[tlines_id_].first;
		}
		else{
			// Reshape according to the first image of each batch
			// on single input batches allows for inputs of varying dimension.
			cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				new_height, new_width, is_color);
			CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
		}

		// Use data_transformer to infer the expected blob shape from a cv_img.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();

		if (use_gps){
			Dtype* prefetch_gps_lat = batch->gps_lat_.mutable_cpu_data();
			Dtype* prefetch_gps_lon = batch->gps_lon_.mutable_cpu_data();

			for (int item_id = 0; item_id < batch_size; ++item_id) {
				// get a blob
				timer.Start();

				CHECK_GT(tlines_size, tlines_id_);
				cv::Mat cv_img = ReadImageToCVMat(root_folder + tlines_[tlines_id_].first,
					new_height, new_width, is_color);
				CHECK(cv_img.data) << "Could not load " << tlines_[tlines_id_].first;


				read_time += timer.MicroSeconds();
				timer.Start();
				// Apply transformations (mirror, crop...) to the image
				int offset = batch->data_.offset(item_id);
				this->transformed_data_.set_cpu_data(prefetch_data + offset);
				this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
				trans_time += timer.MicroSeconds();


				prefetch_gps_lat[item_id] = tlines_[tlines_id_].second;
				prefetch_gps_lon[item_id] = tlines_[tlines_id_].third;
				// go to the next iter
				tlines_id_++;
				if (tlines_id_ >= tlines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					tlines_id_ = 0;
					if (this->layer_param_.image_data_param().shuffle()) {
						ShuffleImagesTri();
					}
				}
			}

		}
		else
		{
			Dtype* prefetch_label = batch->label_.mutable_cpu_data();
			for (int item_id = 0; item_id < batch_size; ++item_id) {
				// get a blob
				timer.Start();

				CHECK_GT(lines_size, lines_id_);
				cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
					new_height, new_width, is_color);
				CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;


				read_time += timer.MicroSeconds();
				timer.Start();
				// Apply transformations (mirror, crop...) to the image
				int offset = batch->data_.offset(item_id);
				this->transformed_data_.set_cpu_data(prefetch_data + offset);
				this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
				trans_time += timer.MicroSeconds();



				prefetch_label[item_id] = lines_[lines_id_].second;
				// go to the next iter
				lines_id_++;
				if (lines_id_ >= lines_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					lines_id_ = 0;
					if (this->layer_param_.image_data_param().shuffle()) {
						ShuffleImages();
					}
				}

			}
		}

		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(ImageDataLayer);
	REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
