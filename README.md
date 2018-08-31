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

Original dataset by Chen et al. (2011): https://purl.stanford.edu/vn158kj2087

1. Training query images

   1.1 Flickr Images

      Flickr id's available at [sf_flickr.txt](https://github.com/hyojinie/crn/blob/master/sf_flickr.txt) 
   
      Each pair in the file consists of the saved image name and the corresponding Flickr id

      Currently, original images are accessible by http://flickr.com/photo.gne?id=(Put Photo Id Here)

      Please refer to copyrights of each images. We plan to provide scripts for downloading images later.

   1.2 Google Streetview Research Dataset

      Available at [ICMLA'11 Streetview Recognition Challenge](http://www.icmla-conference.org/icmla11/challenge.htm)

2. Reference images 

   Available at https://purl.stanford.edu/vn158kj2087

3. Test query images

   Available at https://purl.stanford.edu/vn158kj2087


# Training

** Important Details for Training on New Datasets (described in the paper) **
Step 1> Train the base representation (e.g. NetVLAD) first.
Step 2> Jointly train CRN + the base representation.
In this way, CRN is trained in a more stable manner + yields better performance.

0. Install the custom Caffe & PyCaffe (Includes custom layers built for this method)

1. Download dataset and perform pre-processing on query images (cropping of training queries to three square patches: {left, center, right} or {top, center, bottom} based on the aspect ratio of the original image. The patches should be named as [OriginalName]_aux1.jpg, [OriginalName].jpg, and [OriginalName]_aux2.jpg, respectively.) (Todo: provide script)

2. Image data lists are available at https://www.dropbox.com/s/qv2qkzd4vx25wqm/data.zip?dl=0 

   This contains
   
   1.1 lists of triplets used for training, validation 
   
      - training: all_sanfran_netvlad_trn_fr.txt
      
      - validation: val_sanfran_netvlad_trn_fr.txt
   
   1.2 lists of images for feature extraction used for evaluation
   
      - query: sanfran_q3_featext_fr.txt
      
      - reference: sanfran_sv_featext_fr.txt
   
   * Note: The subfolder "download" is depreciated
   
   * Adjust file paths before starting training

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

* Current version was tested on Ubuntu14.04 with CUDA 7. For Ubuntu16.04 with CUDA 8, please follow the instructions at https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
