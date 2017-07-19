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
# Models

# Training

# Learning rate scheduling hacks 
Learning rate scheduling is done based on the behavior of the training loss (babysitting). The authors did not use the scheduling as specified in the solver.prototxt, but whenever the training loss reached a plateau, learning rate was reduced by gamma (as specified in the solver.prototxt). 

# Evaluation

