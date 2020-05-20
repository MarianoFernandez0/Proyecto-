# Generated with SMOP  0.41
from libsmop import *
# play_segmentation_version2.m

    
@function
def play_segmentation(filename=None,*args,**kwargs):
    varargin = play_segmentation.varargin
    nargin = play_segmentation.nargin

    # Plays the segmented video file results
    
    # Load the segmentation video mat file
    load(filename)
    # Play Detection Movie
    figure(1)
    hold('on')
    grid('on')
    axis(concat([0,640,0,480]))
    # Number of frames
    video_length,__=size(fieldnames(measurements),nargout=2)
# play_segmentation_version2.m:12
    for k in arange(1,video_length).reshape(-1):
        frame=sprintf('frame%d',k)
# play_segmentation_version2.m:16
        h=plot(getattr(measurements,(frame)).x,getattr(measurements,(frame)).y,'r+','MarkerSize',12)
# play_segmentation_version2.m:19
        pause(0.05)
        if (k < video_length):
            delete(concat([h(1)]))
    