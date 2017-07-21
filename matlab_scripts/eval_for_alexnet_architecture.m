%% load gt information
load('sv_cartoid.mat');
load('sv_name_GPS_Sanfran.mat');
addpath('Evaluation_Package');
groundTruth = readGroundTruthFile('Evaluation_Package/cartoid_groundTruth_2014_04.txt', 803);
groundTruth = groundTruth';

%% Subsample data for PCA
% It takes too much memory to load descriptors from all images
% at once, so we divide the street views into three sets
numSV = 1062468; % number of street views 
thirdSV = numSV/3;
assert(thirdSV - floor(thirdSV) == 0);

% subsample images for PCA
file = fopen([init_file_path '/' sv_binary_file_name '.bin'],'r');
cat_ref_p = fread(file, [4096*4, thirdSV], 'float=>single');
cat_ref_p = cat_ref_p';
sampled_cat_ref_p1 = cat_ref_p(1:2:end,:);
clear cat_ref_p1;
sampled_cat_ref_p1 = normr(sampled_cat_ref_p1);

cat_ref_p2 = fread(file, [4096*4, thirdSV], 'float=>single');
cat_ref_p2 = cat_ref_p2';
sampled_cat_ref_p2 = cat_ref_p2(1:2:end,:);
clear cat_ref_p2;
sampled_cat_ref_p2 = normr(sampled_cat_ref_p2);

cat_ref_p3 = fread(file, [4096*4, thirdSV], 'float=>single');
cat_ref_p3 = cat_ref_p3';
sampled_cat_ref_p3 = cat_ref_p3(1:2:end,:);
clear cat_ref_p3;
sampled_cat_ref_p3 = normr(sampled_cat_ref_p3);

x = vertcat(sampled_cat_ref_p1, sampled_cat_ref_p2, sampled_cat_ref_p3);

clear sampled_cat_ref_p1
clear sampled_cat_ref_p2
clear sampled_cat_ref_p3
clear start_pt end_pt n_divisor i_div
fclose(file);

%% PCA 
mu_x = mean(x, 1);
x =  bsxfun(@minus, x, mu_x);
Cx = x'*x;
Cx = Cx/size(x, 1);
[Ux,Sx,Vx] = svd(Cx);
Nx = diag(1./sqrt(diag(Sx) + 1e-9));
Uwx = Ux*Nx;
Utmux = mu_x*Uwx;
%
save([test_path '/pcaWhiteR'],'Uwx','Utmux','-v7.3');
%% PCA to half dimension (8192)
clear x;
clear mapx;
clear Nx;
clear Vx;
clear Cx;
clear Sx;

Utrc = Uwx(:,1:4096*2);

fileW = fopen([test_path '/' sv_binary_file_name '_8192.bin'],'w');
fileO = fopen([init_file_path '/' sv_binary_file_name '.bin'],'r');

% Again, we divide the street views into three sets for memory
% 1st set
cat_ref_p = fread(fileO, [4096*4, thirdSV], 'float=>single');
cat_ref_p =  normc(cat_ref_p);
cat_ref_p = Utrc'*cat_ref_p;
cat_ref_p = bsxfun(@plus, cat_ref_p, (-Utmux(:,1:4096*2))');
fwrite(fileW, cat_ref_p,'float');
% 2nd set
cat_ref_p = fread(fileO, [4096*4, thirdSV], 'float=>single');
cat_ref_p =  normc(cat_ref_p);
cat_ref_p = Utrc'*cat_ref_p;
cat_ref_p = bsxfun(@plus, cat_ref_p, (-Utmux(:,1:4096*2))');
fwrite(fileW, cat_ref_p,'float');
% 3rd set
cat_ref_p = fread(fileO, [4096*4, thirdSV], 'float=>single');
cat_ref_p =  normc(cat_ref_p);
cat_ref_p = Utrc'*cat_ref_p;
cat_ref_p = bsxfun(@plus, cat_ref_p, (-Utmux(:,1:4096*2))');
fwrite(fileW, cat_ref_p,'float');

clear cat_ref_p

fclose(fileW);
fclose(fileO);
%%
file = fopen([test_path '/' sv_binary_file_name '_8192.bin'],'r');
cat_ref = fread(file, [4096*2, Inf], 'float=>single');
fclose(file);
cat_ref = cat_ref';
cat_ref = cat_ref(1:1062468,:);
% l2 normalize 
n_divisor = 4; % divide into four sets for mermory 
for i_div = 1:n_divisor
    start_pt = 265617*(i_div-1)+1;
    end_pt = 265617*i_div;
    cat_ref(start_pt:end_pt,:) = normr(cat_ref(start_pt:end_pt,:));
end
clear start_pt end_pt n_divisor i_div
%%
file = fopen([init_file_path '/' q3_binary_file_name '.bin'],'r');
query_ref = fread(file, [4096*4, Inf], 'float=>single');
fclose(file);
query_ref = query_ref';

query_ref = query_ref*Utrc;
query_ref = bsxfun(@plus, query_ref, -Utmux(:,1:4096*2));
file = fopen([test_path '/' q3_binary_file_name '_8192.bin'],'w');
fwrite(file,query_ref','float');
fclose(file);

% evaluate
query_des = normr(query_ref); % l2 normalize
validate_common;

save([test_path '/' 'test_w_pcaR_8192'],'ret_res','plot_res');