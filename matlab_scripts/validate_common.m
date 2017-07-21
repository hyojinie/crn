ret_res = zeros(size(numel(groundTruth),1),1);
ret_bidx = uint32( zeros( 1000, numel(groundTruth) ) );

score_sv_pc = cell(3,1);
for i_parts = 1:3
    % for sf dataset, three 480x480 crops from 480x640 image is combined
    % each corresponds to leftmost, center, rightmost crops
    reweight_des_p = query_des(i_parts:3:end,:);
    
    % similarity as dot product as image descriptors are l2 normalized
    score_sv_pc{i_parts,1} = cat_ref*reweight_des_p';
end

for t_iter = 1:numel(groundTruth)
    disp(num2str(t_iter));
    this_gt_list = groundTruth{t_iter,1};
    
    score_sv = zeros(size(cat_ref,1),1);
    
    % combining results from three crops mentioned above
    for i_parts = 1:3        
        score_sv_p = score_sv_pc{i_parts,1}(:, t_iter);
        score_sv = score_sv + score_sv_p;
    end
    
    [bestval,bestidx] = sort(score_sv, 'descend');
    ret_bidx(:,t_iter) = uint32( bestidx(1:1000,1) );
    
    shortlist = sv_cartoid(bestidx(1:100),1);
    bIsGT = ismember(shortlist, this_gt_list);
    idxFoundGT = find(bIsGT>0);
    
    if isempty(idxFoundGT)
        ret_res(t_iter,1) = 0;
    else
        ret_res(t_iter,1) = min(idxFoundGT);
    end
end
% plot recall
plot_res = compute_cumgraph_rank(ret_res);
figure; plot(plot_res); title(datestr(now));