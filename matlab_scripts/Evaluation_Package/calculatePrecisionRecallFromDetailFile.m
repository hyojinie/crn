function data = calculatePrecisionRecallFromDetailFile(detailFile, correct, numQueryImages, numTopMatches)

fid = fopen(detailFile, 'r');
if fid <= 0
    error('Cannot read: %s', detailFile);
end
postGVThreshold = 0:5:100;
numCorrect = zeros(size(postGVThreshold));
numUnknown = zeros(size(postGVThreshold));
numImagesRead = 0;
for nImage = 1:numQueryImages
%     disp(nImage);
    [queryImageName, count] = fscanf(fid, '%s', 1);
    if count <= 0
        break;
    end
    numImagesRead = numImagesRead + 1;
    
    for nTop = 1:numTopMatches
        numFeatureMatches = fscanf(fid, '%d', 1);
        dummy = fscanf(fid, '%d', 1);
        databaseImageName = fscanf(fid, '%s', 1);
        if nTop == 1
            numFeatureMatchesTop = numFeatureMatches;
            correctTop = correct(nImage,nTop);
        end
    end % nTop
    for nThreshold = 1:numel(postGVThreshold)
        if numFeatureMatchesTop < postGVThreshold(nThreshold)
            numUnknown(nThreshold) = numUnknown(nThreshold) + 1;
        elseif correctTop == 1
            numCorrect(nThreshold) = numCorrect(nThreshold) + 1;
        end
    end % nThreshold
    
end % nImage
disp(['Processed ' num2str(numImagesRead) ' query images']);
fclose(fid);

data.recall = numCorrect / numQueryImages;
data.precision = numCorrect ./ (numQueryImages - numUnknown);