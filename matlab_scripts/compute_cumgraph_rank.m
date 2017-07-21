function cumgraph_rank = compute_cumgraph_rank(test_rank)
% compute recall
cumgraph_rank = zeros(1,100);
for iter_q = 1:numel(test_rank)
    if test_rank(iter_q,1) == 0
        perQ_cumgraph_rank = zeros(1,100);
    else 
        perQ_cumgraph_rank = ones(1,100);
        c_bins = 1:numel(perQ_cumgraph_rank);
        perQ_cumgraph_rank(c_bins<test_rank(iter_q,1)) = 0;
    end
    cumgraph_rank = cumgraph_rank + perQ_cumgraph_rank;
end