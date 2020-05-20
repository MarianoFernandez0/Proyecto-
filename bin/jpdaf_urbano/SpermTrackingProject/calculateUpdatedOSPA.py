# Generated with SMOP  0.41
from libsmop import *
# calculateUpdatedOSPA.m

    
@function
def calculateUpdatedOSPA(zTotal=None,TrackRecord=None,*args,**kwargs):
    varargin = calculateUpdatedOSPA.varargin
    nargin = calculateUpdatedOSPA.nargin

    OSPA.p = copy(2)
# calculateUpdatedOSPA.m:3
    OSPA.l = copy(25)
# calculateUpdatedOSPA.m:4
    OSPA.c = copy(50)
# calculateUpdatedOSPA.m:5
    # OSPA.p = 2;
# OSPA.l = 25;
# OSPA.c = 50;
    
    # OSPA.p = 2;
# OSPA.l = 50;
# OSPA.c = 50;
    
    # Number of Frames
    numFrames=length(unique(zTotal(3,arange())))
# calculateUpdatedOSPA.m:17
    # Number of True Tracks
    trueTracks=unique(zTotal(4,arange()))
# calculateUpdatedOSPA.m:20
    m=length(trueTracks)
# calculateUpdatedOSPA.m:21
    # Number of Estimated Tracks
    estTracks=unique(TrackRecord(arange(),1))
# calculateUpdatedOSPA.m:24
    n=length(estTracks)
# calculateUpdatedOSPA.m:25
    # /////////////////////////////////////////////////////////////////////// #
    
    #   Assign Labels
    
    # /////////////////////////////////////////////////////////////////////// #
    
    # Distance Cut-off
    DELTA=100
# calculateUpdatedOSPA.m:36
    # Initialize the Distance Matrix
    D=zeros(m,n)
# calculateUpdatedOSPA.m:39
    # For Each True Track
    for i in arange(1,m).reshape(-1):
        # True Track Measured Position
        x=zTotal(concat([arange(1,3)]),(zTotal(4,arange()) == i))
# calculateUpdatedOSPA.m:45
        # x = zTotal([5 6 3], (zTotal(4,:) == trueTracks(i)));
        # For Each Estimated Track
        for j in arange(1,n).reshape(-1):
            # Estimated Track j position x y and time k
        # y = TrackRecord((TrackRecord(:,1) == estTracks(j)), [19 20 4])';
            # Measurements
            y=TrackRecord((TrackRecord(arange(),1) == j),concat([5,6,4])).T
# calculateUpdatedOSPA.m:57
            for k in arange(1,numFrames).reshape(-1):
                # Find the time index in X and Y corresponding to now
                x_idx=find(x(3,arange()) == k)
# calculateUpdatedOSPA.m:63
                y_idx=find(y(3,arange()) == k)
# calculateUpdatedOSPA.m:64
                if logical_not(isempty(x_idx)) and logical_not(isempty(y_idx)):
                    # If both tracks exist, calculate their gated distance
                    d=norm(x(arange(1,2),x_idx) - y(arange(1,2),y_idx))
# calculateUpdatedOSPA.m:70
                    D[i,j]=D(i,j) + min(DELTA,d)
# calculateUpdatedOSPA.m:71
                else:
                    # Non-matching
                    D[i,j]=D(i,j) + DELTA
# calculateUpdatedOSPA.m:76
            # Average distance over all frames
            D[i,j]=D(i,j) / numFrames
# calculateUpdatedOSPA.m:83
    
    # Label the estimated and GT tracks
    label_matrix=munkres(D)
# calculateUpdatedOSPA.m:90
    # Calculate OSPA
    for k in arange(1,numFrames).reshape(-1):
        # The set of ground truth tracks at time k
        X=[]
# calculateUpdatedOSPA.m:96
        X_label=[]
# calculateUpdatedOSPA.m:99
        for i in arange(1,m).reshape(-1):
            # Ground Truth Track i
            x=zTotal(arange(1,3),(zTotal(4,arange()) == trueTracks(i)))
# calculateUpdatedOSPA.m:104
            x_idx=find(x(3,arange()) == k)
# calculateUpdatedOSPA.m:107
            if logical_not(isempty(x_idx)):
                X=concat([X,x(arange(1,2),x_idx)])
# calculateUpdatedOSPA.m:111
                X_label=concat([X_label,i])
# calculateUpdatedOSPA.m:112
        # The set of estimated tracks at time k
        Y=[]
# calculateUpdatedOSPA.m:119
        Y_label=[]
# calculateUpdatedOSPA.m:122
        for j in arange(1,n).reshape(-1):
            # Estimated Track j 
        # y = TrackRecord((TrackRecord(:,1) == j), [19:20 4])';
            y=TrackRecord((TrackRecord(arange(),1) == estTracks(j)),concat([5,6,4])).T
# calculateUpdatedOSPA.m:128
            y_idx=find(y(3,arange()) == k)
# calculateUpdatedOSPA.m:131
            if logical_not(isempty(y_idx)):
                Y=concat([Y,y(arange(1,2),y_idx)])
# calculateUpdatedOSPA.m:136
                iy=find(label_matrix(arange(),j))
# calculateUpdatedOSPA.m:137
                if logical_not(isempty(iy)):
                    Y_label=concat([Y_label,iy])
# calculateUpdatedOSPA.m:140
                else:
                    Y_label=concat([Y_label,999999])
# calculateUpdatedOSPA.m:142
        # Calculate OSPA-T distance
        d_ospa(k),eps_loc(k),eps_card(k)=trk_ospa_dist(X,X_label,Y,Y_label,OSPA,nargout=3)
# calculateUpdatedOSPA.m:151
    