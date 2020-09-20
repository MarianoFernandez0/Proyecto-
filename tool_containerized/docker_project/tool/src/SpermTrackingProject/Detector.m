% /////////////////////////////////////////////////////////////////////////
%
%   Sperm Segmentation Processing
%
%   Leonardo F. Urbano
%
%   April 4th, 2015
%
% /////////////////////////////////////////////////////////////////////////

function Detector(dataFile, videoFile, numFrames)

pkg load video
pkg load image

tic


% /////////////////////////////////////////////////////////////////////////
%
%   Process the Video
%
% /////////////////////////////////////////////////////////////////////////

% Load Video
video = VideoReader(videoFile);

% Display the waitbar
hWaitbar = waitbar(0, 'Processing ...');

Z = [];


h1 = fspecial('gaussian', 11, 1);
%h2 = fspecial('log', 9, 0.3);
h2 = fspecial('log', 10, 0.3);

% Process Each Frame
for k = 1:numFrames;
    %currFrame = rgb2gray(read(video, k));
    currFrame = rgb2gray(readFrame(video, k)); %adaptado a octave
    I = currFrame;
    %figure; imshow(currFrame) %%%
    
    % Top-hat filter
    I = I - imtophat(imcomplement(I), strel('ball', 5, 5));
    %figure; imshow(imcomplement(I));
    
    % Repeat gaussian filter    
    for jj = 1:5
        I = imfilter(I, h1);
    end
    %figure; imshow(I);  %%%
        
    I = imfilter(I, h2);
    %figure; imshow(I);
    %figure; imshow(imcomplement(I));
               
    bw = im2bw(I, 1.1*graythresh(I));
    % figure; imshow(bw); %%%
    
    bw2 = imclearborder(bw);
    bw2 = imclose(bw2, strel('disk', 1, 0)); %en octave hay que agrar el "0"
    % figure; imshow(bw2); %%%
    
    bw3 = imdilate(imerode(bw2, strel('diamond', 2)), strel('diamond', 1));    
    %figure; imshow(bw3); %%%
    
    % Label the blobs
    [labelMatrix, ~] = bwlabel(bw3, 8);
    d = regionprops(labelMatrix, 'Centroid');
    g = cat(1, d.Centroid);
    x = g(:,1);
    y = g(:,2);
    
    % Exclude objects smaller than 5 pixels
    d = regionprops(labelMatrix, 'Area');
    g = cat(1, d.Area);
    idx = (g>=5);
    
    bigCellThresh = 30;
    bigIdx = (g>=bigCellThresh);
    
    %figure; imshow(bw3); hold on;
    %plot(x(bigIdx), y(bigIdx), 'r+');
    

    
    
    % Raw data
    xdata = x(idx)';
    ydata = y(idx)';
    
    % Segmentation Results
    % figure; imshow(currFrame); hold on; plot(xdata, ydata, 'r+', 'MarkerSize', 10)
        
    % Save the Detections to the Z structure
    Z = [Z [xdata; ydata; k * ones(1,length(xdata))]];
    
    % Update the waitbar
    
    waitbar(k/numFrames, hWaitbar);
    
end

close(hWaitbar);

% Save the Segmentation Results
csvwrite(dataFile, Z)



%     xdata = [];
%     ydata = [];
%     edgePixels = 5;
%     for jj = 1:length(x)
%
%         if (y(jj) < (480-edgePixels)) ...
%                 && (y(jj) > edgePixels) ...
%                 && (x(jj) < (640-edgePixels)) ...
%                 && (x(jj) > edgePixels)
%
%             xdata = [xdata x(jj)];
%             ydata = [ydata y(jj)];
%
%         end
%
%     end

% toc
% px2um = 0.857
% um2px = 1/px2um
% % Draw Scale Bar
% hRect = rectangle('Position', [20 460 100*um2px 4*um2px]);
% set(hRect, 'FaceColor', 'k', 'EdgeColor', 'k');
