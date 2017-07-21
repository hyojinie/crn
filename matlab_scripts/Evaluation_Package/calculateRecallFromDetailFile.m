function recall = calculateRecallFromDetailFile(detailFile, correct, GVThreshold, numImages, numTopMatches)

fid = fopen(detailFile, 'r');
if fid <= 0
    error('Cannot read: %s', detailFile);
end
recall = zeros(1, numTopMatches);
numImagesRead = 0;
for nImage = 1:numImages
%     disp(nImage);
    [queryImageName, count] = fscanf(fid, '%s', 1);
    if count <= 0
        break
    end
    numImagesRead = numImagesRead + 1;
    
    alreadyCorrect = 0;
    for nTop = 1:numTopMatches
        numFeatureMatches = fscanf(fid, '%d', 1);
        dummy = fscanf(fid, '%d', 1);
        databaseImageName = fscanf(fid, '%s', 1);
        if alreadyCorrect == 1
            recall(nTop) = recall(nTop) + 1;
        elseif (correct(nImage,nTop) == 1) && (numFeatureMatches >= GVThreshold)
            alreadyCorrect = 1;
            recall(nTop) = recall(nTop) + 1;
        end
    end % nTop
    
end % nImage
disp(['Processed ' num2str(numImagesRead) ' query images']);
fclose(fid);
recall = recall / numImages;