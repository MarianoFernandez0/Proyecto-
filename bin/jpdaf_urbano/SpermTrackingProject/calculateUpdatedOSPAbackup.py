# Generated with SMOP  0.41
from libsmop import *
# calculateUpdatedOSPAbackup.m

    
@function
def calculateUpdatedOSPA(zTotal=None,TrackRecord=None,*args,**kwargs):
    varargin = calculateUpdatedOSPA.varargin
    nargin = calculateUpdatedOSPA.nargin

    # OSPA.p = 1;
# OSPA.l = 25;
# OSPA.c = 50;
    
    OSPA.p = copy(2)
# calculateUpdatedOSPAbackup.m:7
    OSPA.l = copy(25)
# calculateUpdatedOSPAbackup.m:8
    OSPA.c = copy(50)
# calculateUpdatedOSPAbackup.m:9
    # OSPA.p = 2;
# OSPA.l = 50;
# OSPA.c = 50;
    
    # Number of Frames
    numFrames=length(unique(zTotal(3,arange())))
# calculateUpdatedOSPAbackup.m:17
    # Number of True Tracks
    trueTracks=unique(zTotal(4,arange()))
# calculateUpdatedOSPAbackup.m:20
    m=length(trueTracks)
# calculateUpdatedOSPAbackup.m:21
    # Number of Estimated Tracks
    estTracks=unique(TrackRecord(arange(),1))
# calculateUpdatedOSPAbackup.m:24
    n=length(estTracks)
# calculateUpdatedOSPAbackup.m:25
    # /////////////////////////////////////////////////////////////////////// #
    
    #   Assign Labels
    
    # /////////////////////////////////////////////////////////////////////// #
    
    # Distance Cut-off
    DELTA=100
# calculateUpdatedOSPAbackup.m:36
    # Initialize the Distance Matrix
    D=zeros(m,n)
# calculateUpdatedOSPAbackup.m:39
    # For Each True Track
    for i in arange(1,m).reshape(-1):
        # True Track Measured Position
    # x = zTotal([1:3], (zTotal(4,:) == i));
        # True Track i True Position and time
        x=zTotal(concat([5,6,3]),(zTotal(4,arange()) == trueTracks(i)))
# calculateUpdatedOSPAbackup.m:48
        for j in arange(1,n).reshape(-1):
            # Estimated Track j position x y and time k
            y=TrackRecord((TrackRecord(arange(),1) == estTracks(j)),concat([19,20,4])).T
# calculateUpdatedOSPAbackup.m:54
            # y = TrackRecord((TrackRecord(:,1) == j), [5 6 4])';
            # For Each Frame
            for k in arange(1,numFrames).reshape(-1):
                # Find the time index in X and Y corresponding to now
                x_idx=find(x(3,arange()) == k)
# calculateUpdatedOSPAbackup.m:63
                y_idx=find(y(3,arange()) == k)
# calculateUpdatedOSPAbackup.m:64
                if logical_not(isempty(x_idx)) and logical_not(isempty(y_idx)):
                    d=norm(x(arange(1,2),x_idx) - y(arange(1,2),y_idx))
# calculateUpdatedOSPAbackup.m:68
                    D[i,j]=D(i,j) + min(DELTA,d)
# calculateUpdatedOSPAbackup.m:69
                else:
                    D[i,j]=D(i,j) + DELTA
# calculateUpdatedOSPAbackup.m:71
            # Average distance over all frames
            D[i,j]=D(i,j) / numFrames
# calculateUpdatedOSPAbackup.m:77
    
    # Label the estimated and GT tracks
    label_matrix=munkres(D)
# calculateUpdatedOSPAbackup.m:84
    # Calculate OSPA
    for k in arange(1,numFrames).reshape(-1):
        # The set of ground truth tracks at time k
        X=[]
# calculateUpdatedOSPAbackup.m:90
        X_label=[]
# calculateUpdatedOSPAbackup.m:93
        for i in arange(1,m).reshape(-1):
            # Ground Truth Track i
            x=zTotal(arange(1,3),(zTotal(4,arange()) == trueTracks(i)))
# calculateUpdatedOSPAbackup.m:98
            x_idx=find(x(3,arange()) == k)
# calculateUpdatedOSPAbackup.m:101
            if logical_not(isempty(x_idx)):
                X=concat([X,x(arange(1,2),x_idx)])
# calculateUpdatedOSPAbackup.m:105
                X_label=concat([X_label,i])
# calculateUpdatedOSPAbackup.m:106
        # The set of estimated tracks at time k
        Y=[]
# calculateUpdatedOSPAbackup.m:113
        Y_label=[]
# calculateUpdatedOSPAbackup.m:116
        for j in arange(1,n).reshape(-1):
            # Estimated Track j 
        # y = TrackRecord((TrackRecord(:,1) == j), [19:20 4])';
            y=TrackRecord((TrackRecord(arange(),1) == estTracks(j)),concat([5,6,4])).T
# calculateUpdatedOSPAbackup.m:122
            y_idx=find(y(3,arange()) == k)
# calculateUpdatedOSPAbackup.m:125
            if logical_not(isempty(y_idx)):
                Y=concat([Y,y(arange(1,2),y_idx)])
# calculateUpdatedOSPAbackup.m:130
                iy=find(label_matrix(arange(),j))
# calculateUpdatedOSPAbackup.m:131
                if logical_not(isempty(iy)):
                    Y_label=concat([Y_label,iy])
# calculateUpdatedOSPAbackup.m:134
                else:
                    Y_label=concat([Y_label,999999])
# calculateUpdatedOSPAbackup.m:136
        # Calculate OSPA-T distance
        d_ospa(k),eps_loc(k),eps_card(k)=trk_ospa_dist(X,X_label,Y,Y_label,OSPA,nargout=3)
# calculateUpdatedOSPAbackup.m:145
    