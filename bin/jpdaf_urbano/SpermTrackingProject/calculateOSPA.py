# Generated with SMOP  0.41
from libsmop import *
# calculateOSPA.m

    
@function
def calculateOSPA(zTotal=None,TrackRecord=None,*args,**kwargs):
    varargin = calculateOSPA.varargin
    nargin = calculateOSPA.nargin

    OSPA.p = copy(1)
# calculateOSPA.m:3
    OSPA.l = copy(25)
# calculateOSPA.m:4
    OSPA.c = copy(50)
# calculateOSPA.m:5
    # Number of Frames
    numFrames=length(unique(zTotal(3,arange())))
# calculateOSPA.m:8
    # Number of True Tracks
    m=length(unique(zTotal(4,arange())))
# calculateOSPA.m:11
    # Number of Estimated Tracksplot
    n=length(unique(TrackRecord(arange(),1)))
# calculateOSPA.m:14
    # /////////////////////////////////////////////////////////////////////// #
    
    #   Assign Labels
    
    # /////////////////////////////////////////////////////////////////////// #
    
    # Distance Cut-off
    DELTA=100
# calculateOSPA.m:25
    # Initialize the Distance Matrix
    D=zeros(m,n)
# calculateOSPA.m:28
    # For Each True Track
    for i in arange(1,m).reshape(-1):
        # True Track Measured Position
    # x = zTotal([1:3], (zTotal(4,:) == i));
        # True Track i True Position and time
        x=zTotal(concat([5,6,3]),(zTotal(4,arange()) == i))
# calculateOSPA.m:37
        for j in arange(1,n).reshape(-1):
            # Estimated Track j
            y=TrackRecord((TrackRecord(arange(),1) == j),concat([arange(19,20),4])).T
# calculateOSPA.m:43
            # y = TrackRecord((TrackRecord(:,1) == j), [5 6 4])';
            # For Each Video Frame
            for k in arange(1,numFrames).reshape(-1):
                # Find the time index in xt and yt corresponding to now
                x_idx=find(x(3,arange()) == k)
# calculateOSPA.m:52
                y_idx=find(y(3,arange()) == k)
# calculateOSPA.m:53
                if logical_not(isempty(x_idx)) and logical_not(isempty(y_idx)):
                    d=norm(x(arange(1,2),x_idx) - y(arange(1,2),y_idx))
# calculateOSPA.m:57
                    D[i,j]=D(i,j) + min(DELTA,d)
# calculateOSPA.m:58
                else:
                    D[i,j]=D(i,j) + DELTA
# calculateOSPA.m:60
            # Average distance over all frames
            D[i,j]=D(i,j) / numFrames
# calculateOSPA.m:65
    
    # Label the estimated and GT tracks
    label_matrix=munkres(D)
# calculateOSPA.m:72
    # Calculate OSPA
    for k in arange(1,numFrames).reshape(-1):
        X=[]
# calculateOSPA.m:77
        X_label=[]
# calculateOSPA.m:78
        for i in arange(1,m).reshape(-1):
            # Ground Truth Track i
            x=zTotal(arange(1,3),(zTotal(4,arange()) == i))
# calculateOSPA.m:83
            x_idx=find(x(3,arange()) == k)
# calculateOSPA.m:86
            if logical_not(isempty(x_idx)):
                X=concat([X,x(arange(1,2),x_idx)])
# calculateOSPA.m:90
                X_label=concat([X_label,i])
# calculateOSPA.m:91
        Y=[]
# calculateOSPA.m:96
        Y_label=[]
# calculateOSPA.m:97
        for j in arange(1,n).reshape(-1):
            # Estimated Track j 
        # y = TrackRecord((TrackRecord(:,1) == j), [19:20 4])';
            y=TrackRecord((TrackRecord(arange(),1) == j),concat([5,6,4])).T
# calculateOSPA.m:103
            y_idx=find(y(3,arange()) == k)
# calculateOSPA.m:106
            if logical_not(isempty(y_idx)):
                Y=concat([Y,y(arange(1,2),y_idx)])
# calculateOSPA.m:111
                iy=find(label_matrix(arange(),j))
# calculateOSPA.m:112
                if logical_not(isempty(iy)):
                    Y_label=concat([Y_label,iy])
# calculateOSPA.m:115
                else:
                    Y_label=concat([Y_label,99999])
# calculateOSPA.m:117
        # Calculate OSPA-T distance
        d_ospa(k),eps_loc(k),eps_card(k)=trk_ospa_dist(X,X_label,Y,Y_label,OSPA,nargout=3)
# calculateOSPA.m:126
    