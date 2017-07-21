#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layers/rank_hard_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


using std::max;

using namespace std;
using namespace cv;

// hyo
// #define pi 3.14159265358979323846
#define earthRadiusKm 6371.0


namespace caffe {

int myrandom (int i) { return caffe_rng_rand()%i;}

/**
* Returns the distance between two points on the Earth.
* Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
* @return The distance between the two points in kilometers
*/
double distanceEarth(double lat1r, double lon1r, double lat2r, double lon2r) {
	// lat and lon should be in radians!
	// image_data_layer reads degrees and convert them in radians to save in gps_lat & gps_lon
	double u, v;
	u = sin((lat2r - lat1r) / 2);
	v = sin((lon2r - lon1r) / 2);
	return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  diff_.ReshapeLike(*bottom[0]);
  dis_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  mask_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	int hard_num = neg_num * hard_ratio;
	int rand_num = neg_num * rand_ratio;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	// const Dtype* label = bottom[1]->cpu_data();
	const Dtype* gps_lat = bottom[1]->cpu_data();
	const Dtype* gps_lon = bottom[2]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < num * num; i ++)
	{
		dis_data[i] = 0;
		mask_data[i] = 0;
	}

	// calculate distance
	for(int i = 0; i < num; i ++)
	{
		for(int j = i + 1; j < num; j ++)
		{
			const Dtype* fea1 = bottom_data + i * dim;
			const Dtype* fea2 = bottom_data + j * dim;
			Dtype ts = 0;
			for(int k = 0; k < dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}
			dis_data[i * num + j] = -ts;
			dis_data[j * num + i] = -ts;
		}
	}

	//select samples

	vector<pair<float, int> >negpairs;
	vector<int> sid1;
	vector<int> sid2;
	vector<pair<float, float> >gps_pairs_neg;
	// do masking. masked ones are used as negatives to make triplets.
	// mask stores negatives for each pair of positives
	for(int i = 0; i < num; i += pair_size)
	{
		negpairs.clear();
		gps_pairs_neg.clear();
		sid1.clear();
		sid2.clear();
		for(int j = 0; j < num; j ++)
		{
			Dtype tloss_hn = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			// Note: [i * num + j] is equal to [i,j]
			if (tloss_hn == 0) continue;

			// distanceEarth(double lat1r, double lon1r, double lat2r, double lon2r)
			double dist_earth = distanceEarth(gps_lat[j], gps_lon[j], gps_lat[i], gps_lon[i]);

			if ((j != i + 2) && (j != i + 3)){

				if ((dist_earth < 0.225)) {
					continue; // distance less than 225 meters
				}

				std::vector< pair<float, float> >::iterator it;
				bool duplicated_pano = false;
				int index_it = 0;
				for (it = gps_pairs_neg.begin(); it != gps_pairs_neg.begin(); it++){
					if (it->first == gps_lat[j] && it->second == gps_lon[j]){
						duplicated_pano = true;
						if (negpairs[index_it].first > dis_data[i * num + j]){
							negpairs[index_it].first = dis_data[i * num + j];
							negpairs[index_it].second = j;
						}
					}
					index_it++;
				}

				if (duplicated_pano){
					continue;
				}
			}

			//if (dist_earth < 0.225)
			//	continue; // distance less than 225 meters
			// hyo : This loss is only used for hard negative mining 
			// (with margin = 0.2, no additional pos pair term) to avoid collapsing of model f(x) = 0.
			// Dtype tloss_hn = max(Dtype(0.0), dis_data[i * num + i + 1] * Dtype(margin + 1.00) - dis_data[i * num + j] + Dtype(margin));			

			negpairs.push_back(make_pair(dis_data[i * num + j], j));
			gps_pairs_neg.push_back(make_pair(gps_lat[j], gps_lon[j]));
		}
		if(negpairs.size() <= neg_num)
		{
			for(int j = 0; j < negpairs.size(); j ++)
			{
				int id = negpairs[j].second;
				mask_data[i * num + id] = 1;
			}
			continue;
		}
		sort(negpairs.begin(), negpairs.end());

		for(int j = 0; j < neg_num; j ++)
		{
			sid1.push_back(negpairs[j].second); 
			// top hard negatives (size: neg_num)
		}
		for(int j = neg_num; j < negpairs.size(); j ++)
		{
			sid2.push_back(negpairs[j].second); 
			// other hard negatives 
		}
		std::random_shuffle(sid1.begin(), sid1.end(), myrandom); // shuffle top hard negatives. why? (over fitting?)
		for(int j = 0; j < min(hard_num, (int)(sid1.size()) ); j ++)
		{
			mask_data[i * num + sid1[j]] = 1;
		}
		for(int j = hard_num; j < sid1.size(); j ++)
		{
			sid2.push_back(sid1[j]); // add left over top hard negatives to sid2
		}
		std::random_shuffle(sid2.begin(), sid2.end(), myrandom); // shuffle sid2
		for(int j = 0; j < min( rand_num, (int)(sid2.size()) ); j ++)
		{
			mask_data[i * num + sid2[j]] = 1; // random sample negatives from sid
		}

	}


}




template <typename Dtype>
void RankHardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	// const Dtype* label = bottom[1]->cpu_data();
	const Dtype* gps_lat = bottom[1]->cpu_data();
	const Dtype* gps_lon = bottom[2]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();


	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();      // 4
	int pair_size = rank_param.pair_size();  // 5
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	set_mask(bottom);
	Dtype loss = 0;
	int cnt = neg_num * num / pair_size * 2; 
	// Here, * 2 is used to average tloss1 and tloss2.
	// (neg_num x batch_size / pair_size) is the number of positive pairs (batch_size / pair_size) times neg_num,
	// so, it is the number of triplets times 2.

	for(int i = 0; i < num; i += pair_size)
	{
		for(int j = 0; j < num; j ++)
		{
			if(mask_data[i * num + j] == 0) continue;
			// hyo - new ranking loss
			//Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + i + 1] * Dtype(margin + 1.0) - dis_data[i * num + j] + Dtype(margin)) + dis_data[i * num + i + 1] + Dtype(1.0);
			Dtype tloss1 = max(Dtype(0.0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			// D(i, i+1) - D(i,j) --> D(i, p) - D(i, n)
			//Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + i + 1] * Dtype(margin + 1.0) - dis_data[(i + 1) * num + j] + Dtype(margin)) + dis_data[i * num + i + 1] + Dtype(1.0);
			Dtype tloss2 = max(Dtype(0.0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));
			// D(i, i+1) - D(i+1,j) --> D(i, p) - D(p, n) : Easy way of increasing training samples?
			loss += tloss1 + tloss2;
		}
	}

	// hyo - divide by 2 (as in contrastive loss)
	loss = loss / cnt; // / Dtype(2)
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	const Dtype* bottom_data = bottom[0]->cpu_data();
	// const Dtype* label = bottom[1]->cpu_data();
	
	// const Dtype* gps_lat = bottom[1]->cpu_data();
	// const Dtype* gps_lon = bottom[2]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();


	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	// float hard_ratio = rank_param.hard_ratio();
	// float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < count; i ++ )
		bottom_diff[i] = 0;

	int cnt = neg_num * num / pair_size * 2;

	for(int i = 0; i < num; i += pair_size)
	{
		const Dtype* fori = bottom_data + i * dim;
	    const Dtype* fpos = bottom_data + (i + 1) * dim;

	    Dtype* fori_diff = bottom_diff + i * dim;
		Dtype* fpos_diff = bottom_diff + (i + 1) * dim;
		for(int j = 0; j < num; j ++)
		{
			if(mask_data[i * num + j] == 0) continue;
			// hyo new loss
			Dtype tloss1_bk = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			Dtype tloss2_bk = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));

			const Dtype* fneg = bottom_data + j * dim;
			Dtype* fneg_diff = bottom_diff + j * dim;
			if(tloss1_bk > 0)
			{
				for(int k = 0; k < dim; k ++)
			    {
					fori_diff[k] += (fneg[k] - fpos[k]); // / (pairNum * 1.0 - 2.0);
					fpos_diff[k] += -fori[k]; // / (pairNum * 1.0 - 2.0);
					fneg_diff[k] +=  fori[k];
			    }
			}
			if(tloss2_bk > 0)
			{
				for(int k = 0; k < dim; k ++)
				{
					fori_diff[k] += -fpos[k]; // / (pairNum * 1.0 - 2.0);
					fpos_diff[k] += fneg[k] - fori[k]; // / (pairNum * 1.0 - 2.0);
				    fneg_diff[k] += fpos[k];
				}
			}
		}
	}

	// hyo - divide by 2 (as in contrastive loss)
	for (int i = 0; i < count; i ++)
	{
		bottom_diff[i] = bottom_diff[i] / cnt; // / Dtype(2)
	}

}

#ifdef CPU_ONLY
STUB_GPU(RankHardLossLayer);
#endif

INSTANTIATE_CLASS(RankHardLossLayer);
REGISTER_LAYER_CLASS(RankHardLoss);

}  // namespace caffe
