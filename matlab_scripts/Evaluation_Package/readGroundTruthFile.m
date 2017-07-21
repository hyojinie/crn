function groundTruth = readGroundTruthFile(fileName, numQueryImages)

fid = fopen(fileName, 'r');
if fid <= 0
    error('Cannot read: %s', fileName);
end
numImagesRead = 0;
groundTruth = cell(1, numQueryImages);
for nImage = 1:numQueryImages
    tline = fgetl(fid);
    if length(tline) == 0
        break;
    end
    numImagesRead = numImagesRead + 1;
    
    [token, remain] = strtok(tline);
    gids = {};
    while length(remain) > 0
        [token, remain] = strtok(remain);
        gids{end+1} = token;
    end % while
    groundTruth{nImage} = gids;
end % nImage
disp(['Found ground truth for ' num2str(numImagesRead) ' query images']);
fclose(fid);
