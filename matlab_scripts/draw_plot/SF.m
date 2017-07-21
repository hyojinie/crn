x_axis = [1:5 10:5:50];
    
figure;
hold on;
load('crn_netvlad_vgg_fullres\test_w_pcaR_16384.mat');
plot(x_axis, 100*plot_res(x_axis)/803, 'ro-');
load('netvlad_vgg_fullres\test_w_pcaR_16384.mat');
plot(x_axis, 100*plot_res(x_axis)/803, 'rx-');
 load('crn_netvlad_alex_fullres\test_w_pcaR_16384.mat');
plot(x_axis, 100*plot_res(x_axis)/803, 'bo-');
load('netvlad_alex_fullres\test_w_pcaR_16384.mat');
plot(x_axis, 100*plot_res(x_axis)/803, 'bx-');

%set(findall(gcf,'type','axes'),'fontsize',11)
legend('Ours (VGG16)', 'NetVLAD fine-tuned (VGG16)', 'Ours (Alexnet)', 'NetVLAD fine-tuned (Alexnet)');

ylabel('Recall (%)','fontsize',12)
xlabel('N top retrievals','fontsize',12)

set(gcf,'Color','w');

ylim([73, 89]); % in camera ready