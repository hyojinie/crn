Evaluation package for San Francisco Landmark Dataset
David Chen (dmchen@stanford.edu)
Department of Electrical Engineering
Stanford University

---------------------------------------------------------
Contents of evaluation package
---------------------------------------------------------
cartoid_groundTruth.txt : ground truth file
*.detail : sample retrieval results files
*.m : Matlab/Octave scripts for calculating recall and precision

---------------------------------------------------------
Format of retrieval results file
---------------------------------------------------------
Each retrieval result file should have the following format. A couple of *.detail files are provided in 
this evaluation package that follow this format exactly and can be used to reproduce the PCI results in 
our CVPR 2011 paper. Please note that the database image filenames should contain the corresponding 
building IDs, e.g., 671172716, for comparison against the ground truth building IDs.

<name of query image file #1>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #1>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #2>
...
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #50>

<name of query image file #2>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #1>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #2>
...
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #50>

...
<name of query image file #803>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #1>
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #2>
...
<number of RANSAC inliers> <dummy indicator> <name of retrieved database image file #50>

---------------------------------------------------------
Evaluation scripts
---------------------------------------------------------

Please use the script "plotSanFranciscoRetrievalResults.m" to read the included ground truth file and 
retrieval results files, calculate recall and precision, and generate plots of the recall and 
precision. At the top of this script, you can adjust parameters like GVThreshold (threshold for 
RANSAC inliers) and topMatches (number of retrieved database images in a ranked shortlist) as well as 
specify different retrieval results files for your experiments.
