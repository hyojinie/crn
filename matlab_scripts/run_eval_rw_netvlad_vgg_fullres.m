clear all;
%% create folder for saving result
dirSave = 'eval_result';
mkdir(dirSave);
test_path = [dirSave '/' 'sanfran_test/rw_netvlad_vgg_fullres'];
mkdir(test_path);
%% directory where extracted features are stored (in binary format)
dir_features_binary = '../extracted_features/rw_netvlad_vgg_fullres';
sv_binary_file_name = 'sanfran_sv_rw_netvlad_vgg';
q3_binary_file_name = 'sanfran_q3_rw_netvlad_vgg';
%% run evaluation
eval_for_vgg_architecture;