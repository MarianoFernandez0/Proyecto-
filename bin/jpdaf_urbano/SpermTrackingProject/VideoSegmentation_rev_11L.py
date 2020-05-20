# Generated with SMOP  0.41
from smop.libsmop import *
import cv2
# VideoSegmentation_rev_11L.m

    # /////////////////////////////////////////////////////////////////////////
    
    #   Sperm Segmentation Processing
    
    #   Leonardo F. Urbano
    
    #   April 4th, 2015
    
    # /////////////////////////////////////////////////////////////////////////
#    clear('all')
#    close_('all')
#    font_size=12
# VideoSegmentation_rev_11L.m:10
    # Choose file to Segment
# [fileName, pathName] = uigetfile('*.mp4', 'Choose a video')
# videoFile = fullfile(pathName, fileName);

videoFile='/Users/EXAMPLE/Desktop/EXAMPLE.mp4'
# VideoSegmentation_rev_11L.m:19
tic
# /////////////////////////////////////////////////////////////////////////

#   Process the Video

# /////////////////////////////////////////////////////////////////////////

# Load Video
#video=VideoReader(videoFile)
video = cv2.VideoCapture(videoFile)
# VideoSegmentation_rev_11L.m:32
# Number of frames to process (1.5 minutes)
numFrames=dot(60,15)
# VideoSegmentation_rev_11L.m:35
# Display the waitbar
# hWaitbar = waitbar(0, 'Processing ...');

Z=[]
# VideoSegmentation_rev_11L.m:40
h1=fspecial('gaussian',11,1)
# VideoSegmentation_rev_11L.m:43
h2=fspecial('log',9,0.3)
# VideoSegmentation_rev_11L.m:44
# Process Each Frame
for k in arange(1,numFrames).reshape(-1):
    # currFrame = rgb2gray(read(video, k))
    _, img = video.read()
    currFrame=rgb2gray(img)
# VideoSegmentation_rev_11L.m:49
    I=copy(currFrame)
# VideoSegmentation_rev_11L.m:51
    # Top-hat filter
    I=I - imtophat(imcomplement(I),strel('ball',5,5))
# VideoSegmentation_rev_11L.m:55
    # Repeat gaussian filter
    for jj in arange(1,5).reshape(-1):
        I=imfilter(I,h1)
# VideoSegmentation_rev_11L.m:60
    #figure; imshow(I);  ###
    I=imfilter(I,h2)
# VideoSegmentation_rev_11L.m:64
    #figure; imshow(imcomplement(I));
    bw=im2bw(I,dot(1.1,graythresh(I)))
# VideoSegmentation_rev_11L.m:68
    bw2=imclearborder(bw)
# VideoSegmentation_rev_11L.m:71
    bw2=imclose(bw2,strel('disk',1))
# VideoSegmentation_rev_11L.m:72
    bw3=imdilate(imerode(bw2,strel('diamond',2)),strel('diamond',1))
# VideoSegmentation_rev_11L.m:75
    # Label the blobs
    labelMatrix,__=bwlabel(bw3,8,nargout=2)
# VideoSegmentation_rev_11L.m:79
    d=regionprops(labelMatrix,'Centroid')
# VideoSegmentation_rev_11L.m:80
    g=cat(1,d.Centroid)
# VideoSegmentation_rev_11L.m:81
    x=g(arange(),1)
# VideoSegmentation_rev_11L.m:82
    y=g(arange(),2)
# VideoSegmentation_rev_11L.m:83
    d=regionprops(labelMatrix,'Area')
# VideoSegmentation_rev_11L.m:86
    g=cat(1,d.Area)
# VideoSegmentation_rev_11L.m:87
    idx=(g >= 5)
# VideoSegmentation_rev_11L.m:88
    bigCellThresh=30
# VideoSegmentation_rev_11L.m:90
    bigIdx=(g >= bigCellThresh)
# VideoSegmentation_rev_11L.m:91
    #plot(x(bigIdx), y(bigIdx), 'r+');
    # Raw data
    xdata=x(idx).T
# VideoSegmentation_rev_11L.m:100
    ydata=y(idx).T
# VideoSegmentation_rev_11L.m:101
    # figure; imshow(currFrame); hold on; plot(xdata, ydata, 'r+', 'MarkerSize', 10)
    # Save the Detections to the Z structure
    Z=concat([Z,concat([[xdata],[ydata],[dot(k,ones(1,length(xdata)))]])])
# VideoSegmentation_rev_11L.m:107
    k / numFrames

# close(hWaitbar);

# Save the Segmentation Results
csvwrite(fullfile(concat([videoFile,'_MergedMeasurementDataFile.dat'])),Z)
#     xdata = [];
#     ydata = [];
#     edgePixels = 5;
#     for jj = 1:length(x)

#         if (y(jj) < (480-edgePixels)) ...
#                 && (y(jj) > edgePixels) ...
#                 && (x(jj) < (640-edgePixels)) ...
#                 && (x(jj) > edgePixels)

#             xdata = [xdata x(jj)];
#             ydata = [ydata y(jj)];

#         end

#     end

# toc
# px2um = 0.857
# um2px = 1/px2um
# # Draw Scale Bar
# hRect = rectangle('Position', [20 460 100*um2px 4*um2px]);
# set(hRect, 'FaceColor', 'k', 'EdgeColor', 'k');