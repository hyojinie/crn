clear all;
%% create folder for saving result
dirSave = 'eval_result';
mkdir(dirSave);
test_path = [dirSave '/' 'sanfran_test/rw_netvlad_alexnet_fullres'];
mkdir(test_path);
%% directory where extracted features are stored (in binary format)
dir_features_binary = '../extracted_features/rw_netvlad_alexnet_fullres';
sv_binary_file_name = 'sanfran_sv_rw_netvlad_alexnet';
q3_binary_file_name = 'sanfran_q3_rw_netvlad_alexnet';
%% run evaluation
eval_for_alexnet_architecture;