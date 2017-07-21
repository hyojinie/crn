#!/usr/bin/env sh
 
TOOLS=./build/tools

THISFOLDER=crn_cvpr17/rw_net

$TOOLS/caffe train --weights=crn_cvpr17/netvlad/snapshots/netvlad_vgg.caffemodel --solver=$THISFOLDER/solver_rw_vgg.prototxt --gpu=0





