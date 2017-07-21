THISROOTPATH=crn_cvpr17/rw_net
MODELNAME=snapshots/rw_netvlad_vgg

SAVENAME=sanfran_sv_rw_netvlad_vgg_lmdb
SAVEBIN=sanfran_sv_rw_netvlad_vgg.bin

rm -rf $THISROOTPATH/$SAVENAME

./build/tools/extract_features.bin  $THISROOTPATH/$MODELNAME.caffemodel $THISROOTPATH/get_rw_vgg_sv_fullres.prototxt vlad_postL2 $THISROOTPATH/$SAVENAME 48294 lmdb GPU 1

python lmdb_to_binary_pycaffe/feat2binary_pycaffe_fast.py -i $THISROOTPATH/$SAVENAME -o $THISROOTPATH/$SAVEBIN

SAVENAME=sanfran_q3_rw_netvlad_vgg_lmdb
SAVEBIN=sanfran_q3_rw_netvlad_vgg.bin

rm -rf $THISROOTPATH/$SAVENAME

./build/tools/extract_features.bin  $THISROOTPATH/$MODELNAME.caffemodel $THISROOTPATH/get_rw_vgg_q3_fullres.prototxt vlad_postL2 $THISROOTPATH/$SAVENAME 803 lmdb GPU 1

python lmdb_to_binary_pycaffe/feat2binary_pycaffe_fast.py -i $THISROOTPATH/$SAVENAME -o $THISROOTPATH/$SAVEBIN

echo "Done."

