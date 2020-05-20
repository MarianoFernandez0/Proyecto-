# Generated with SMOP  0.41
from libsmop import *
# plotTrackHistory_4L.m

    
@function
def plotTrackHistory_4L(TrackRecord=None,T=None,startTime=None,endTime=None,*args,**kwargs):
    varargin = plotTrackHistory_4L.varargin
    nargin = plotTrackHistory_4L.nargin

    # How many tracks are there?
    trackList=unique(TrackRecord(arange(),1)).T
# plotTrackHistory_4L.m:4
    # Plot each track
    for trk in trackList.reshape(-1):
        # Get data for this track
        dataIdx=find(TrackRecord(arange(),1) == trk)
# plotTrackHistory_4L.m:10
        posX=TrackRecord(dataIdx,5)
# plotTrackHistory_4L.m:11
        posY=TrackRecord(dataIdx,6)
# plotTrackHistory_4L.m:12
        measX=TrackRecord(dataIdx,19)
# plotTrackHistory_4L.m:13
        measY=TrackRecord(dataIdx,20)
# plotTrackHistory_4L.m:14
        time=multiply(TrackRecord(dataIdx,4),T)
# plotTrackHistory_4L.m:15
        v=sqrt((diff(measX) / T) ** 2 + (diff(measY) / T) ** 2)
# plotTrackHistory_4L.m:18
        timeIdx=find(time >= logical_and(startTime,time) <= endTime)
# plotTrackHistory_4L.m:21
        plot(measX(timeIdx),measY(timeIdx),'k.')
        plot(measX(timeIdx),measY(timeIdx),'k')
    