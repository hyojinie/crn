# Learned Contextual Feature Reweighting for Image Geo-Localization 

This code is developed based on [Caffe](http://caffe.berkeleyvision.org)

This code is the implementation for the network with the context-based feature reweighting in the paper:

[Hyo Jin Kim](http://hyojin.web.unc.edu), [Enrique Dunn](http://enrique.web.unc.edu), and [Jan-Michael Frahm](http://frahm.web.unc.edu). "Learned Contextual Feature Reweighting for Image Geo-Localization". Proc. of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kim_Learned_Contextual_Feature_CVPR_2017_paper.pdf)] [[project page](http://hyojin.web.unc.edu/crn/)]

If you use our codes or models in your research, please cite:
```txt
@inproceedings{kim2017crn,
  title={Learned Contextual Feature Reweighting for Image Geo-Localization},
  author={Kim, Hyo Jin and Dunn, Enrique and  Frahm, Jan-Michael},
  booktitle={CVPR},
  year= {2017}
}
```

# Dataset for San Francisco Benchmark
1. Training queries

   1.1 Flickr Images

   Flickr id's available at [sf_flickr.txt](https://github.com/hyojinie/crn/blob/master/sf_flickr.txt)

   Currently, original images are accessible by http://flickr.com/photo.gne?id=(Put Photo Id Here)

   Please refer to copyrights of each images. We plan to provide scripts for downloading images later.

   1.2 Google Streetview Research Dataset

   Available at [ICMLA'11 Challenge]http://www.icmla-conference.org/icmla11/challenge.htm

   Please use the dataset for the purposes of research only and Google don't allow any commercial use of the dataset. Let us know if you    have problem downloading them. 

2. Reference Dataset
Original dataset by Chen et al. (2011): https://purl.stanford.edu/vn158kj2087

   ROI cropped versions used for training: Available on request. Please specify your name, affiliation, and purpose for the dataset to hyojin(at)cs.unc.edu.

3. Test queries
Available at https://purl.stanford.edu/vn158kj2087


# Training

0. Install the custom Caffe & pycaffe (Includes custom layers built for this method)

1. Download dataset & do pre-processing (cropping of training queries right-center-left or top-center-bottom) (Todo: provide script)

2. Image data lists are available at https://www.dropbox.com/s/qv2qkzd4vx25wqm/data.zip?dl=0 

   This contains
   
   1.1 lists of triplets used for training, validation 
   
      - training: all_sanfran_netvlad_trn_fr.txt
      
      - validation: val_sanfran_netvlad_trn_fr.txt
   
   1.2 lists of images for feature extraction used for evaluation
   
      - query: sanfran_q3_featext_fr.txt
      
      - reference: sanfran_sv_featext_fr.txt
   
   * Note: the subfolder "download" is depreciated
   
   * Adjust file paths before start training

3. Run 
```txt
   cd crn/caffe   
   crn_cvpr17/rw_net/train_rw_alex.sh # for alexnet-based network
   crn_cvpr17/rw_net/train_rw_vgg.sh # for vgg16-based network
```

3.1. Fine-tuning of NetVLAD on SF benchmark
```txt
   crn_cvpr17/netvlad/train_netvlad_alex.sh # for alexnet-based network
   crn_cvpr17/netvlad/train_netvlad_alex.sh # for vgg16-based network
```
# Pre-Trained Models
We provide pre-trained models in [trained_models](https://github.com/hyojinie/crn/tree/master/trained%20_models)

# Evaluation
Evaluation scripts available at [matlab_scripts](https://github.com/hyojinie/crn/tree/master/matlab_scripts)

1. Evaluation of CRN+NetVLAD
```txt
run_eval_rw_netvlad_alexnet_fullres.m	
run_eval_rw_netvlad_vgg_fullres.m
```

2. Evaluation of NetVLAD
```txt
run_eval_netvlad_alexnet_fullres.m
run_eval_netvlad_vgg_fullres.m	
```

# Misc
* Learning rate scheduling: Learning rate scheduling is done through babysitting. Whenever the training loss reached a plateau, learning rate was reduced by gamma (as specified in the solver.prototxt).

