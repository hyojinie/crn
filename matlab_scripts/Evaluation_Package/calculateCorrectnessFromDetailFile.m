function correct = calculateCorrectnessFromDetailFile(detailFile, groundTruth, numQueryImages, numTopMatches)

fid = fopen(detailFile, 'r');
if fid <= 0
    error('Cannot read: %s', detailFile);
end
correct = zeros(numQueryImages, numTopMatches);
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
        
        foundGround = 0;
        for nGround = 1:length(groundTruth{nImage})
            foundGround = ~isempty(strfind(databaseImageName, groundTruth{nImage}{nGround}));
            if foundGround
                break;
            end
        end % nGround
        correct(nImage, nTop) = foundGround;
    end % nTop
    
end % nImage
disp(['Processed ' num2str(numImagesRead) ' query images']);
fclose(fid);