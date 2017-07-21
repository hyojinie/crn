#!/usr/bin/env sh
 
TOOLS=./build/tools

THISFOLDER=crn_cvpr17/rw_net

$TOOLS/caffe train --weights=crn_cvpr17/netvlad/snapshots/netvlad_alex.caffemodel --solver=$THISFOLDER/solver_rw_alex.prototxt --gpu=0






