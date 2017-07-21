% plotSanFrancisoRetrievalResults
clc; clear all; close all;

GVThreshold = 30;
topMatches = 50;
numQueryImages = 803;

% Define retrieval result files
noRerankDetailFile = 'SVKNNLoxelSingle2RTree-D6-Navteq-SF-PCI-March2011-HistEq-Upright-SIFT-300m.queryResult.accuracy.label.detail';
rerankDetailFile = 'SVKNNLoxelSingle2RTree-D6-Navteq-SF-PCI-March2011-HistEq-Upright-SIFT-300m.queryResult.accuracy.label.rerank.detail';

% Read ground truth file
groundTruth = readGroundTruthFile('cartoid_groundTruth.txt', numQueryImages);

% Determine correct-ness of retrieval results
correctMarch2011PCILoxelSingle2RD6SIFT300m = ...
    calculateCorrectnessFromDetailFile(noRerankDetailFile, groundTruth, numQueryImages, topMatches);
correctMarch2011PCILoxelSingle2RD6SIFT300mRerank = ...
    calculateCorrectnessFromDetailFile(rerankDetailFile, groundTruth, numQueryImages, topMatches);

% Compute recall and precision
recallMarch2011PCILoxelSingle2RD6SIFT300m = ...
    calculateRecallFromDetailFile(noRerankDetailFile, correctMarch2011PCILoxelSingle2RD6SIFT300m, GVThreshold, numQueryImages, topMatches);
recallMarch2011PCILoxelSingle2RD6SIFT300mRerank = ...
    calculateRecallFromDetailFile(rerankDetailFile, correctMarch2011PCILoxelSingle2RD6SIFT300mRerank, GVThreshold, numQueryImages, topMatches);
PRMarch2011PCILoxelSingle2RD6SIFT300mRerank = ...
    calculatePrecisionRecallFromDetailFile(rerankDetailFile, correctMarch2011PCILoxelSingle2RD6SIFT300mRerank, numQueryImages, topMatches);

% Plot results
figure(1); clf;
set(gcf, 'Position', [100 80 350 300]);
figure(2); clf;
set(gcf, 'Position', [500 80 350 300]);
figure(1);
skip = 3;
range = [1:skip:topMatches-skip+1 topMatches];
topMatchesVec = 1:topMatches;
h = plot(...
    topMatchesVec(range), 100* recallMarch2011PCILoxelSingle2RD6SIFT300mRerank(range), 'k-s', ...
    topMatchesVec(range), 100* recallMarch2011PCILoxelSingle2RD6SIFT300m(range), 'k--s'); 
grid on;
set(h, 'LineWidth', 2);
set(gca, 'FontSize', 10);
set(gca, 'XTick', 0:10:50);
xlabel('Number of Top Database Candidates');
ylabel('Recall (Percent)');
axis([0 topMatches+1 0 70]);
figure(2);
range = 1:numel(PRMarch2011PCILoxelSingle2RD6SIFT300mRerank.precision);
h = plot(...
    100*(1 - PRMarch2011PCILoxelSingle2RD6SIFT300mRerank.precision(range)), 100*PRMarch2011PCILoxelSingle2RD6SIFT300mRerank.recall(range), 'k-s');
grid on;
set(h, 'LineWidth', 2);
set(gca, 'FontSize', 10);
xlabel('100% Minus Precision (Percent)'); ylabel('Recall (Percent)');
axis([0 100 0 100]);
