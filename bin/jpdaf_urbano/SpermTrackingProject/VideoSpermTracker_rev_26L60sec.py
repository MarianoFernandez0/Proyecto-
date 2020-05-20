# Generated with SMOP  0.41
from smop.libsmop import *
# VideoSpermTracker_rev_26L60sec.m

    # /////////////////////////////////////////////////////////////////////// #

    #   JPDA Multi-Target Tracking Algorithm

    #   By  Leonardo F. Urbano

    #   April 11th, 2015


    # /////////////////////////////////////////////////////////////////////// #
#    clear('all')
#    close_('all')
#    tic

#s=RandStream('mt19937ar','Seed',1)
# VideoSpermTracker_rev_26L60sec.m:13
#RandStream.setGlobalStream(s)
# dbstop if warning
# dbstop if error

# Video A
dataFile='/Users/EXAMPLE/Desktop/EXAMPLE.mp4_NewNewDataFile.dat'
# VideoSpermTracker_rev_26L60sec.m:20
videoFile='/Users/EXAMPLE/Desktop/EXAMPLE.mp4'
# VideoSpermTracker_rev_26L60sec.m:21
# Load Data File
zTotal=csvread(dataFile)
# VideoSpermTracker_rev_26L60sec.m:24
# Number of frames to analyze (60 sec x 15 fps = 900)
# numFrames = 10 * 15; # length(unique(zTotal(3,:)));
numFrames=dot(60,15)
# VideoSpermTracker_rev_26L60sec.m:28
# Load Video File
video=VideoReader(videoFile)
# VideoSpermTracker_rev_26L60sec.m:31
# Multi-Target Tracking Algorithm
# 1 = NN
# 2 = GNN
# 3 = PDAF
# 4 = JPDAF
# 5 = ENN-JPDAF
# 6 = Iterated Multi-assignment
mttAlgorithm=2
# VideoSpermTracker_rev_26L60sec.m:41
# Plot Results
plotResults=0
# VideoSpermTracker_rev_26L60sec.m:44
# SaveMovie
saveMovie=1
# VideoSpermTracker_rev_26L60sec.m:47
# Scale Factor (px2micron)
px2um=dot((12 / 28),2)
# VideoSpermTracker_rev_26L60sec.m:50
um2px=1 / px2um
# VideoSpermTracker_rev_26L60sec.m:51
# ROI Size Parameters
ROIx=640
# VideoSpermTracker_rev_26L60sec.m:54
ROIy=480
# VideoSpermTracker_rev_26L60sec.m:55
# Initalize Filter
T=1 / 15
# VideoSpermTracker_rev_26L60sec.m:58

# Dynamical System & Measurement Equation
F=concat([[1,0,T,0],[0,1,0,T],[0,0,1,0],[0,0,0,1]])
# VideoSpermTracker_rev_26L60sec.m:61
G=concat([[T ** 2 / 2,0],[0,T ** 2 / 2],[T,0],[0,T]])
# VideoSpermTracker_rev_26L60sec.m:62
H=concat([[1,0,0,0],[0,1,0,0]])
# VideoSpermTracker_rev_26L60sec.m:63
# Noise Covariance Matrix
N=dot(2 ** 2,eye(2,2))
# VideoSpermTracker_rev_26L60sec.m:66
# CWNA Process Noise
qMat=concat([[T ** 3 / 3,0,T ** 2 / 2,0],[0,T ** 3 / 3,0,T ** 2 / 2],[T ** 2 / 2,0,T,0],[0,T ** 2 / 2,0,T]])
# VideoSpermTracker_rev_26L60sec.m:69
deltaV=20
# VideoSpermTracker_rev_26L60sec.m:70
qIntensity=deltaV ** 2 / T
# VideoSpermTracker_rev_26L60sec.m:71
Q0=dot(qMat,qIntensity)
# VideoSpermTracker_rev_26L60sec.m:72
# Initial covariance matrix
# P0 = [N N./T; N./T 2*N./T^2];
P0=copy(Q0)
# VideoSpermTracker_rev_26L60sec.m:76

# Initial residual covariance matrix
S0=dot(dot(H,P0),H.T) + N
# VideoSpermTracker_rev_26L60sec.m:79
# Statistical Parameters
PG=0.997
# VideoSpermTracker_rev_26L60sec.m:82

PD=0.95
# VideoSpermTracker_rev_26L60sec.m:83

# Expected number of measurements due to clutter per unit area of the
# surveillance space per scan of data
lam_f=1e-06
# VideoSpermTracker_rev_26L60sec.m:87
# Expected number of measurements from new targets per unit area of the
# surveillance space per scan of data
lam_n=1e-05
# VideoSpermTracker_rev_26L60sec.m:91
# Track Management Thresholds
initialScore=log(lam_n / lam_f)
# VideoSpermTracker_rev_26L60sec.m:94
Pdelete=1e-06
# VideoSpermTracker_rev_26L60sec.m:95
Pconfirm=1e-05
# VideoSpermTracker_rev_26L60sec.m:96
threshDelete=log(Pdelete / (1 - Pconfirm))
# VideoSpermTracker_rev_26L60sec.m:97
threshConfirm=log((1 - Pdelete) / Pconfirm) + initialScore
# VideoSpermTracker_rev_26L60sec.m:98
# Position Gate (um)
gx=chi2inv(PG,2)
# VideoSpermTracker_rev_26L60sec.m:101
# Velocity Gate (um/s)
gv=300
# VideoSpermTracker_rev_26L60sec.m:104
# /////////////////////////////////////////////////////////////////////// #

#   Create Analysis Figures

# /////////////////////////////////////////////////////////////////////// #

if (plotResults):
    fHandles[1]=figure
# VideoSpermTracker_rev_26L60sec.m:114
    set(gca,'Box','On')
    axis('equal')
    axis(concat([0,ROIx,0,ROIy]))
    hold('on')
    grid('on')

# /////////////////////////////////////////////////////////////////////// #

#   Main Loop

# /////////////////////////////////////////////////////////////////////// #

# Clear the Track File
TrackFile=[]
# VideoSpermTracker_rev_26L60sec.m:126
# Clear the Track Record
TrackRecord=[]
# VideoSpermTracker_rev_26L60sec.m:129
# Load the waitbar
hWaitbar=waitbar(0,'Processing ...')
# VideoSpermTracker_rev_26L60sec.m:132
# Process Each Frame
for k in arange(1,numFrames).reshape(-1):
    m=0
# VideoSpermTracker_rev_26L60sec.m:138
    n=0
# VideoSpermTracker_rev_26L60sec.m:139
    # Set of Measurements at Frame k assuming PD
    Z=[]
# VideoSpermTracker_rev_26L60sec.m:142
    Z=multiply(zTotal(arange(1,2),(zTotal(3,arange()) == k)),px2um)
# VideoSpermTracker_rev_26L60sec.m:143
    if logical_not(isempty(Z)):
        __,m=size(Z,nargout=2)
# VideoSpermTracker_rev_26L60sec.m:147
        Z[3,arange()]=0
# VideoSpermTracker_rev_26L60sec.m:149
    # List of Active Tracks
    if logical_not(isempty(TrackFile)):
        t_idx=find(TrackFile(arange(),1) > 0)
# VideoSpermTracker_rev_26L60sec.m:154
        n=length(t_idx)
# VideoSpermTracker_rev_26L60sec.m:155
    # /////////////////////////////////////////////////////////////////// #
    #   Tracking Loop
    # /////////////////////////////////////////////////////////////////// #
    if logical_not(isempty(TrackFile)):
        # Indices of all Measurements at Frame k
        m_idx=arange(1,m)
# VideoSpermTracker_rev_26L60sec.m:167
        Xu=[]
# VideoSpermTracker_rev_26L60sec.m:170
        Pu=[]
# VideoSpermTracker_rev_26L60sec.m:170
        Xp=[]
# VideoSpermTracker_rev_26L60sec.m:171
        Pp=[]
# VideoSpermTracker_rev_26L60sec.m:171
        Sp=[]
# VideoSpermTracker_rev_26L60sec.m:171
        Zp=[]
# VideoSpermTracker_rev_26L60sec.m:171
        d=[]
# VideoSpermTracker_rev_26L60sec.m:172
        f=[]
# VideoSpermTracker_rev_26L60sec.m:172
        for t in arange(1,length(t_idx)).reshape(-1):
            # Track Number in the TrackFile
            trk=t_idx(t)
# VideoSpermTracker_rev_26L60sec.m:178
            sigmaN=TrackFile(trk,25)
# VideoSpermTracker_rev_26L60sec.m:181
            Nk[arange(),arange(),t]=dot(sigmaN ** 2,eye(2,2))
# VideoSpermTracker_rev_26L60sec.m:184
            Xu[arange(),t]=TrackFile(trk,arange(4,7)).T
# VideoSpermTracker_rev_26L60sec.m:187
            Pu[arange(),arange(),t]=concat([[TrackFile(trk,arange(8,11))],[TrackFile(trk,concat([9,arange(12,14)]))],[TrackFile(trk,concat([10,13,arange(15,16)]))],[TrackFile(trk,concat([11,14,16,17]))]])
# VideoSpermTracker_rev_26L60sec.m:188
            oldQhat=concat([[TrackFile(trk,arange(26,29))],[TrackFile(trk,concat([27,arange(30,32)]))],[TrackFile(trk,concat([28,31,arange(33,34)]))],[TrackFile(trk,concat([29,32,34,35]))]])
# VideoSpermTracker_rev_26L60sec.m:189
            Xp[arange(),t]=dot(F,Xu(arange(),end()))
# VideoSpermTracker_rev_26L60sec.m:192
            deltaX=Xp(arange(),t) - Xu(arange(),end())
# VideoSpermTracker_rev_26L60sec.m:195
            c1=0.3
# VideoSpermTracker_rev_26L60sec.m:198
            c2=0.5
# VideoSpermTracker_rev_26L60sec.m:199
            c3=0.2
# VideoSpermTracker_rev_26L60sec.m:200
            newQhat=dot(c1,oldQhat) + dot(dot(c2,deltaX),deltaX.T) + dot(c3,Q0)
# VideoSpermTracker_rev_26L60sec.m:201
            TrackFile[end(),26]=newQhat(1,1)
# VideoSpermTracker_rev_26L60sec.m:203
            TrackFile[end(),27]=newQhat(1,2)
# VideoSpermTracker_rev_26L60sec.m:204
            TrackFile[end(),28]=newQhat(1,3)
# VideoSpermTracker_rev_26L60sec.m:205
            TrackFile[end(),29]=newQhat(1,4)
# VideoSpermTracker_rev_26L60sec.m:206
            TrackFile[end(),30]=newQhat(2,2)
# VideoSpermTracker_rev_26L60sec.m:207
            TrackFile[end(),31]=newQhat(2,3)
# VideoSpermTracker_rev_26L60sec.m:208
            TrackFile[end(),32]=newQhat(2,4)
# VideoSpermTracker_rev_26L60sec.m:209
            TrackFile[end(),33]=newQhat(3,3)
# VideoSpermTracker_rev_26L60sec.m:210
            TrackFile[end(),34]=newQhat(3,4)
# VideoSpermTracker_rev_26L60sec.m:211
            TrackFile[end(),35]=newQhat(4,4)
# VideoSpermTracker_rev_26L60sec.m:212
            # Predicted Covariance and Residual Covariance
            Pp[arange(),arange(),t]=dot(dot(F,Pu(arange(),arange(),end())),F.T) + newQhat
# VideoSpermTracker_rev_26L60sec.m:215
            Sp[arange(),arange(),t]=dot(dot(H,Pp(arange(),arange(),end())),H.T) + Nk(arange(),arange(),t)
# VideoSpermTracker_rev_26L60sec.m:216
            sqrtDet2piSp=sqrt(det(dot(dot(2,pi),Sp(arange(),arange(),t))))
# VideoSpermTracker_rev_26L60sec.m:219
            Zp[arange(),t]=dot(H,Xp(arange(),t))
# VideoSpermTracker_rev_26L60sec.m:222
            for j in arange(1,m).reshape(-1):
                # Measurement Residual
                v_jt=Z(arange(1,2),j) - Zp(arange(),t)
# VideoSpermTracker_rev_26L60sec.m:228
                d_jt=dot(v_jt.T / Sp(arange(),arange(),t),v_jt)
# VideoSpermTracker_rev_26L60sec.m:231
                if (d_jt <= gx) and (norm(v_jt) / T <= gv):
                    # Distance between Track t and Measurement j
                    d[j,t]=d_jt
# VideoSpermTracker_rev_26L60sec.m:237
                    Z[3,j]=1
# VideoSpermTracker_rev_26L60sec.m:240
                else:
                    # No Measuremnt in Validation Gates
                    d[j,t]=Inf
# VideoSpermTracker_rev_26L60sec.m:245
                # Gaussian pdf
                f[j,t]=exp(dot(- 0.5,d(j,t))) / sqrtDet2piSp
# VideoSpermTracker_rev_26L60sec.m:250
        # /////////////////////////////////////////////////////////// #
        #   Identify Track Clusters
        # /////////////////////////////////////////////////////////// #
        # Association Matrix
        A=ceil(f)
# VideoSpermTracker_rev_26L60sec.m:265
        mA,nA=size(A,nargout=2)
# VideoSpermTracker_rev_26L60sec.m:266
        if logical_not(isempty(A)):
            # Identify Track Clusters (tracks gated by measurements)
            clusters=identifyClusters(A)
# VideoSpermTracker_rev_26L60sec.m:273
            numClusters=unique(clusters(arange(),1)).T
# VideoSpermTracker_rev_26L60sec.m:276
        else:
            # If there are no measurements at all, then each track is
        # its own cluster
            clusters=concat([(arange(1,length(t_idx))).T,zeros(length(t_idx),1),(arange(1,length(t_idx))).T])
# VideoSpermTracker_rev_26L60sec.m:282
            numClusters=length(t_idx)
# VideoSpermTracker_rev_26L60sec.m:283
        # /////////////////////////////////////////////////////////// #
        #   Proceses Each Track Cluster
        # /////////////////////////////////////////////////////////// #
        # Process Each Cluster
        for c_idx in numClusters.reshape(-1):
            jj_idx=[]
# VideoSpermTracker_rev_26L60sec.m:298
            tt_idx=[]
# VideoSpermTracker_rev_26L60sec.m:299
            jj_idx=unique(clusters(find(clusters(arange(),1) == c_idx),2)).T
# VideoSpermTracker_rev_26L60sec.m:302
            tt_idx=unique(clusters(find(clusters(arange(),1) == c_idx),3)).T
# VideoSpermTracker_rev_26L60sec.m:305
            #   Data Association
            #//////////////////////////////////////////////////////// #
            beta=0
# VideoSpermTracker_rev_26L60sec.m:314
            LR=0
# VideoSpermTracker_rev_26L60sec.m:315
            if (jj_idx > 0):
                # Likelihood Ratio Matrix
                LR=multiply(multiply((lam_f ** - 1),f(jj_idx,tt_idx)),PD)
# VideoSpermTracker_rev_26L60sec.m:321
                if (mttAlgorithm == 1):
                    # Nearest-neighbor
                    beta=nnAssociation(LR)
# VideoSpermTracker_rev_26L60sec.m:326
                else:
                    if (mttAlgorithm == 2):
                        # Global Nearest Neighbor
                        beta=munkres(- log(LR))
# VideoSpermTracker_rev_26L60sec.m:331
                    else:
                        if (mttAlgorithm == 3):
                            # PDAF
                            beta=pdafAssociation(LR,PD,PG)
# VideoSpermTracker_rev_26L60sec.m:336
                        else:
                            if (mttAlgorithm == 4):
                                beta=jpdafAssociation(LR,PD,PG)
# VideoSpermTracker_rev_26L60sec.m:341
                            else:
                                if (mttAlgorithm == 5):
                                    beta=munkres(- log(jpdafAssociation(LR,PD,PG)))
# VideoSpermTracker_rev_26L60sec.m:346
                                else:
                                    if (mttAlgorithm == 6):
                                        # Iterated multi-assignment
                                        beta=imaAlgorithm(LR,PD,PG)
# VideoSpermTracker_rev_26L60sec.m:351
            # /////////////////////////////////////////////////////// #
            #   Track Update
            # /////////////////////////////////////////////////////// #
            # For Each Track tt in the Cluster
            for tt in arange(1,length(tt_idx)).reshape(-1):
                # Prob-weighted Combined Innovation
                V_t=concat([[0],[0]])
# VideoSpermTracker_rev_26L60sec.m:368
                D_t=0
# VideoSpermTracker_rev_26L60sec.m:371
                v_jt=concat([[0],[0]])
# VideoSpermTracker_rev_26L60sec.m:373
                atLeastOneMeasurement=0
# VideoSpermTracker_rev_26L60sec.m:375
                if (jj_idx > 0):
                    atLeastOneMeasurement=1
# VideoSpermTracker_rev_26L60sec.m:380
                    for jj in arange(1,length(jj_idx)).reshape(-1):
                        # Resudial Between Meas jj and Track tt
                        v_jt=Z(arange(1,2),jj_idx(jj)) - Zp(arange(),tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:386
                        V_t=V_t + dot(beta(jj,tt),v_jt)
# VideoSpermTracker_rev_26L60sec.m:389
                        D_t=D_t + dot(beta(jj,tt),(dot(v_jt,v_jt.T)))
# VideoSpermTracker_rev_26L60sec.m:392
                # Probability that None of the Measurements is Correct
                beta0=(1 - dot(PD,PG)) / (1 - dot(PD,PG) + sum(LR(arange(),tt)))
# VideoSpermTracker_rev_26L60sec.m:399
                K=dot(Pp(arange(),arange(),tt_idx(tt)),H.T) / Sp(arange(),arange(),tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:402
                L=eye(4,4) - dot(K,H)
# VideoSpermTracker_rev_26L60sec.m:405
                P_star=dot(dot(L,Pp(arange(),arange(),tt_idx(tt))),L.T) + dot(dot(K,Nk(arange(),arange(),tt_idx(tt))),K.T)
# VideoSpermTracker_rev_26L60sec.m:408
                P_zero=dot(beta0,Pp(arange(),arange(),tt_idx(tt))) + dot((1 - beta0),P_star)
# VideoSpermTracker_rev_26L60sec.m:411
                P_delta=dot(dot(K,(D_t - dot(V_t,V_t.T))),K.T)
# VideoSpermTracker_rev_26L60sec.m:414
                Pu=P_zero + P_delta
# VideoSpermTracker_rev_26L60sec.m:417
                Xu=Xp(arange(),tt_idx(tt)) + dot(K,V_t)
# VideoSpermTracker_rev_26L60sec.m:420
                Zm=dot(H,Xp(arange(),tt_idx(tt))) + V_t
# VideoSpermTracker_rev_26L60sec.m:423
                trk=t_idx(tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:426
                if (norm(V_t) == 0) and (atLeastOneMeasurement == 0):
                    # If no measurement updated this track
                    deltaL=log(1 - PD)
# VideoSpermTracker_rev_26L60sec.m:432
                    # measurement, then delete it
                    if (TrackFile(trk,24) == 0):
                        TrackFile[trk,1]=0
# VideoSpermTracker_rev_26L60sec.m:437
                else:
                    # If a measurement updated the track
                    deltaL=log(dot(dot(lam_f ** - 1,PD),(dot(dot(2,pi),det(Sp(arange(),arange(),tt_idx(tt))))) ** (- 0.5)) - dot(dot(0.5,V_t.T) / Sp(arange(),arange(),tt_idx(tt)),V_t))
# VideoSpermTracker_rev_26L60sec.m:443
                    TrackFile[trk,24]=TrackFile(trk,24) + 1
# VideoSpermTracker_rev_26L60sec.m:448
                # Update Track File
                TrackFile[trk,2]=TrackFile(trk,2) + deltaL
# VideoSpermTracker_rev_26L60sec.m:453
                TrackFile[trk,3]=k
# VideoSpermTracker_rev_26L60sec.m:454
                TrackFile[trk,4]=Xu(1)
# VideoSpermTracker_rev_26L60sec.m:455
                TrackFile[trk,5]=Xu(2)
# VideoSpermTracker_rev_26L60sec.m:456
                TrackFile[trk,6]=Xu(3)
# VideoSpermTracker_rev_26L60sec.m:457
                TrackFile[trk,7]=Xu(4)
# VideoSpermTracker_rev_26L60sec.m:458
                TrackFile[trk,8]=Pu(1,1)
# VideoSpermTracker_rev_26L60sec.m:459
                TrackFile[trk,9]=Pu(1,2)
# VideoSpermTracker_rev_26L60sec.m:460
                TrackFile[trk,10]=Pu(1,3)
# VideoSpermTracker_rev_26L60sec.m:461
                TrackFile[trk,11]=Pu(1,4)
# VideoSpermTracker_rev_26L60sec.m:462
                TrackFile[trk,12]=Pu(2,2)
# VideoSpermTracker_rev_26L60sec.m:463
                TrackFile[trk,13]=Pu(2,3)
# VideoSpermTracker_rev_26L60sec.m:464
                TrackFile[trk,14]=Pu(2,4)
# VideoSpermTracker_rev_26L60sec.m:465
                TrackFile[trk,15]=Pu(3,3)
# VideoSpermTracker_rev_26L60sec.m:466
                TrackFile[trk,16]=Pu(3,4)
# VideoSpermTracker_rev_26L60sec.m:467
                TrackFile[trk,17]=Pu(4,4)
# VideoSpermTracker_rev_26L60sec.m:468
                TrackFile[trk,18]=Zm(1)
# VideoSpermTracker_rev_26L60sec.m:469
                TrackFile[trk,19]=Zm(2)
# VideoSpermTracker_rev_26L60sec.m:470
                TrackFile[trk,20]=max(TrackFile(trk,2) - deltaL,TrackFile(trk,20))
# VideoSpermTracker_rev_26L60sec.m:471
                TrackFile[trk,21]=Sp(1,1,tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:472
                TrackFile[trk,22]=Sp(1,2,tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:473
                TrackFile[trk,23]=Sp(2,2,tt_idx(tt))
# VideoSpermTracker_rev_26L60sec.m:474
                # Update the Track Record for Confirmed Tracks Only
                if (TrackFile(trk,1) > 0):
                    TrackRecord[end() + 1,arange()]=concat([trk,TrackFile(trk,arange())])
# VideoSpermTracker_rev_26L60sec.m:479
        # /////////////////////////////////////////////////////////////// #
        #   Plot the Problem
        # /////////////////////////////////////////////////////////////// #
        if (plotResults == 1):
            pHandles=[]
# VideoSpermTracker_rev_26L60sec.m:496
            currFrame=rgb2gray(read(video,k))
# VideoSpermTracker_rev_26L60sec.m:499
            pHandles[end() + 1]=imshow(currFrame)
# VideoSpermTracker_rev_26L60sec.m:500
            hold('on')
            pHandles[end() + 1]=plot(multiply(Z(1,arange()),um2px),multiply(Z(2,arange()),um2px),'r+')
# VideoSpermTracker_rev_26L60sec.m:503
            pHandles[end() + 1]=plot(multiply(Xp(1,arange()),um2px),multiply(Xp(2,arange()),um2px),'b.')
# VideoSpermTracker_rev_26L60sec.m:504
            ct_idx=find(TrackFile(arange(),1) > 1)
# VideoSpermTracker_rev_26L60sec.m:507
            for t in arange(1,length(ct_idx)).reshape(-1):
                trk=ct_idx(t)
# VideoSpermTracker_rev_26L60sec.m:512
                pHandles[end() + 1]=plotEllipse(multiply(Zp(arange(),t).T,um2px),multiply(dot(gx,Sp(arange(),arange(),t)),um2px ** 2))
# VideoSpermTracker_rev_26L60sec.m:513
                set(pHandles(end()),'Color','r')
                pHandles[end() + 1]=text(multiply(Zp(1,t),um2px) + dot(0.025,ROIx),multiply(Zp(2,t),um2px) + dot(0.025,ROIy),num2str(t_idx(t)),'FontSize',12,'FontWeight','bold','FontName','Arial','Color','k')
# VideoSpermTracker_rev_26L60sec.m:516
            pHandles[end() + 1]=plot(multiply(Z(1,arange()),um2px),multiply(Z(2,arange()),um2px),'r+')
# VideoSpermTracker_rev_26L60sec.m:522
            pause(0.001)
            if (k < numFrames):
                delete(ravel(pHandles))
    # /////////////////////////////////////////////////////////////////// #
    #   Track Promotion / Deletion
    # /////////////////////////////////////////////////////////////////// #
    if logical_not(isempty(TrackFile)):
        t_idx=find(TrackFile(arange(),1) > 0)
# VideoSpermTracker_rev_26L60sec.m:546
        for t in arange(1,length(t_idx)).reshape(-1):
            trk=t_idx(t)
# VideoSpermTracker_rev_26L60sec.m:551
            if (TrackFile(trk,2) > threshConfirm):
                TrackFile[trk,1]=2
# VideoSpermTracker_rev_26L60sec.m:555
            # Change from Maximum Track Score
            scoreDelta=TrackFile(trk,2) - TrackFile(trk,20)
# VideoSpermTracker_rev_26L60sec.m:559
            if (scoreDelta <= threshDelete):
                TrackFile[trk,1]=0
# VideoSpermTracker_rev_26L60sec.m:563
    # /////////////////////////////////////////////////////////////// #
    #   Delete Duplicate Tracks
    # /////////////////////////////////////////////////////////////// #
    if logical_not(isempty(TrackFile)):
        # Set of indices of confirmed tracks
        t_idx=find(TrackFile(arange(),1) > 1)
# VideoSpermTracker_rev_26L60sec.m:579
        if logical_not(isempty(t_idx)) and (length(t_idx) > 1):
            XU=concat([TrackFile(t_idx,4),TrackFile(t_idx,5)])
# VideoSpermTracker_rev_26L60sec.m:583
            for jj in arange(1,length(XU)).reshape(-1):
                DD_idx=concat([t_idx(jj)])
# VideoSpermTracker_rev_26L60sec.m:588
                for tt in arange(1,length(XU)).reshape(-1):
                    DD=sqrt((XU(jj,1) - XU(tt,1)) ** 2 - (XU(jj,2) - XU(tt,2)) ** 2)
# VideoSpermTracker_rev_26L60sec.m:592
                    if (DD > 0) and (DD < 0.01):
                        DD_idx=concat([DD_idx,t_idx(tt)])
# VideoSpermTracker_rev_26L60sec.m:596
                # Which one of the set of redundant tracks has the
            # highest track score?
                DD_idx_max=find(TrackFile(DD_idx,2) == max(TrackFile(DD_idx,2)))
# VideoSpermTracker_rev_26L60sec.m:605
                DD_max=DD_idx(DD_idx_max)
# VideoSpermTracker_rev_26L60sec.m:607
                TracksToDelete=setdiff(DD_idx,DD_max)
# VideoSpermTracker_rev_26L60sec.m:609
                for del_idx in arange(1,length(TracksToDelete)).reshape(-1):
                    TrackFile[TracksToDelete(del_idx),1]=0
# VideoSpermTracker_rev_26L60sec.m:613
    # /////////////////////////////////////////////////////////////////// #
    #   Initiate Tracks on Un-used Measurements
    # /////////////////////////////////////////////////////////////////// #
    if logical_not(isempty(Z)):
        for j in find(Z(3,arange()) == 0).reshape(-1):
            TrackFile[end() + 1,1]=1
# VideoSpermTracker_rev_26L60sec.m:632
            # 1 = Tentative
        # 2 = Confirmed
        # 0 = Deleted
            TrackFile[end(),2]=initialScore
# VideoSpermTracker_rev_26L60sec.m:636
            TrackFile[end(),3]=k
# VideoSpermTracker_rev_26L60sec.m:637
            TrackFile[end(),4]=Z(1,j)
# VideoSpermTracker_rev_26L60sec.m:638
            TrackFile[end(),5]=Z(2,j)
# VideoSpermTracker_rev_26L60sec.m:639
            TrackFile[end(),6]=0
# VideoSpermTracker_rev_26L60sec.m:640
            TrackFile[end(),7]=0
# VideoSpermTracker_rev_26L60sec.m:641
            TrackFile[end(),8]=P0(1,1)
# VideoSpermTracker_rev_26L60sec.m:642
            TrackFile[end(),9]=P0(1,2)
# VideoSpermTracker_rev_26L60sec.m:643
            TrackFile[end(),10]=P0(1,3)
# VideoSpermTracker_rev_26L60sec.m:644
            TrackFile[end(),11]=P0(1,4)
# VideoSpermTracker_rev_26L60sec.m:645
            TrackFile[end(),12]=P0(2,2)
# VideoSpermTracker_rev_26L60sec.m:646
            TrackFile[end(),13]=P0(2,3)
# VideoSpermTracker_rev_26L60sec.m:647
            TrackFile[end(),14]=P0(2,4)
# VideoSpermTracker_rev_26L60sec.m:648
            TrackFile[end(),15]=P0(3,3)
# VideoSpermTracker_rev_26L60sec.m:649
            TrackFile[end(),16]=P0(3,4)
# VideoSpermTracker_rev_26L60sec.m:650
            TrackFile[end(),17]=P0(4,4)
# VideoSpermTracker_rev_26L60sec.m:651
            TrackFile[end(),18]=Z(1,j)
# VideoSpermTracker_rev_26L60sec.m:652
            TrackFile[end(),19]=Z(2,j)
# VideoSpermTracker_rev_26L60sec.m:653
            TrackFile[end(),20]=initialScore
# VideoSpermTracker_rev_26L60sec.m:654
            TrackFile[end(),21]=S0(1,1)
# VideoSpermTracker_rev_26L60sec.m:655
            TrackFile[end(),22]=S0(1,2)
# VideoSpermTracker_rev_26L60sec.m:656
            TrackFile[end(),23]=S0(2,2)
# VideoSpermTracker_rev_26L60sec.m:657
            TrackFile[end(),24]=0
# VideoSpermTracker_rev_26L60sec.m:658
            TrackFile[end(),25]=sqrt(N(1,1))
# VideoSpermTracker_rev_26L60sec.m:659
            TrackFile[end(),26]=Q0(1,1)
# VideoSpermTracker_rev_26L60sec.m:661
            TrackFile[end(),27]=Q0(1,2)
# VideoSpermTracker_rev_26L60sec.m:662
            TrackFile[end(),28]=Q0(1,3)
# VideoSpermTracker_rev_26L60sec.m:663
            TrackFile[end(),29]=Q0(1,4)
# VideoSpermTracker_rev_26L60sec.m:664
            TrackFile[end(),30]=Q0(2,2)
# VideoSpermTracker_rev_26L60sec.m:665
            TrackFile[end(),31]=Q0(2,3)
# VideoSpermTracker_rev_26L60sec.m:666
            TrackFile[end(),32]=Q0(2,4)
# VideoSpermTracker_rev_26L60sec.m:667
            TrackFile[end(),33]=Q0(3,3)
# VideoSpermTracker_rev_26L60sec.m:668
            TrackFile[end(),34]=Q0(3,4)
# VideoSpermTracker_rev_26L60sec.m:669
            TrackFile[end(),35]=Q0(4,4)
# VideoSpermTracker_rev_26L60sec.m:670
    # Track Record Defintion
# TrackRecord(:,1)  Track Number
# TrackRecord(:,2)  Track Rank (0 = deleted, 1 = tentative, 2 = confirmed)
# TrackRecord(:,3)  Track Score
# TrackRecord(:,4)  Frame Number
# TrackRecord(:,5)  Estimated X position at frame k
# TrackRecord(:,6)  Estimated Y position at frame k
# TrackRecord(:,19)  Measured Y position at frame k
# TrackRecord(:,20)  Measured Y position at frame k
    # Update the waitbar
    waitbar(k / numFrames,hWaitbar)

# Close the waitbar
close_(hWaitbar)
# /////////////////////////////////////////////////////////////////////// #

#   Create Movie File

# /////////////////////////////////////////////////////////////////////// #

saveMovie=1
# VideoSpermTracker_rev_26L60sec.m:708
if (saveMovie):
    figure
    set(gca,'Box','On')
    iptsetpref('ImshowBorder','tight')
    # Open the Movie File
    movieFile=fullfile(concat([videoFile,'_Aug23PaperMovie10sec_',num2str(mttAlgorithm)]))
# VideoSpermTracker_rev_26L60sec.m:716
    vidObj=VideoWriter(movieFile)
# VideoSpermTracker_rev_26L60sec.m:717
    set(vidObj,'Quality',100)
    set(vidObj,'FrameRate',1 / T)
    open_(vidObj)
    # Length of Trail History (in frames)
    trailLength=(dot(1,15))
# VideoSpermTracker_rev_26L60sec.m:723
    for k in arange(1,numFrames).reshape(-1):
        # Display the video frame
        imshow(imcomplement(rgb2gray(read(video,k))))
        hold('on')
        set(gcf,'Position',concat([255,90,955,715]))
        numTentativeTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 1),1)))
# VideoSpermTracker_rev_26L60sec.m:734
        numConfirmedTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2),1)))
# VideoSpermTracker_rev_26L60sec.m:737
        timeIdx=find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2)
# VideoSpermTracker_rev_26L60sec.m:742
        trackIdx=unique(TrackRecord(timeIdx,1))
# VideoSpermTracker_rev_26L60sec.m:746
        for trk in trackIdx.T.reshape(-1):
            # Get the indices to the data for this track up to time k
            dataIdx=find(TrackRecord(arange(),1) == logical_and(trk,TrackRecord(arange(),4)) <= k)
# VideoSpermTracker_rev_26L60sec.m:752
            SpMat[1,1]=TrackRecord(dataIdx(end()),22)
# VideoSpermTracker_rev_26L60sec.m:756
            SpMat[1,2]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:757
            SpMat[2,1]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:758
            SpMat[2,2]=TrackRecord(dataIdx(end()),24)
# VideoSpermTracker_rev_26L60sec.m:759
            if (length(dataIdx) <= trailLength):
                posX=multiply(TrackRecord(dataIdx,5),um2px)
# VideoSpermTracker_rev_26L60sec.m:764
                posY=multiply(TrackRecord(dataIdx,6),um2px)
# VideoSpermTracker_rev_26L60sec.m:765
                measX=multiply(TrackRecord(dataIdx,19),um2px)
# VideoSpermTracker_rev_26L60sec.m:766
                measY=multiply(TrackRecord(dataIdx,20),um2px)
# VideoSpermTracker_rev_26L60sec.m:767
            else:
                posX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),5),um2px)
# VideoSpermTracker_rev_26L60sec.m:769
                posY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),6),um2px)
# VideoSpermTracker_rev_26L60sec.m:770
                measX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),19),um2px)
# VideoSpermTracker_rev_26L60sec.m:771
                measY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),20),um2px)
# VideoSpermTracker_rev_26L60sec.m:772
            # Set of Measurements at Frame k assuming PD
            Z=[]
# VideoSpermTracker_rev_26L60sec.m:776
            Z=zTotal(arange(1,2),(zTotal(3,arange()) == k))
# VideoSpermTracker_rev_26L60sec.m:777
            plot(Z(1,arange()),Z(2,arange()),'r+')
            # Plot the Tracks and Measurements up to time k
            plot(posX,posY,'b')
            plot(measX,measY,'g')
            plot(measX,measY,'g.','MarkerSize',5)
            plot(measX(end()),measY(end()),'y+')
            ellipHand=plotEllipse(concat([[posX(end())],[posY(end())]]),multiply(dot(gx,SpMat),um2px ** 2))
# VideoSpermTracker_rev_26L60sec.m:788
            set(ellipHand,'Color','r')
            text(posX(end()) + 5,posY(end()) + 5,num2str(trk),'FontName','Arial','FontSize',12,'FontWeight','Bold','Color','g')
        # Draw 100um scale bar
        hRectangle=rectangle('Position',concat([20,460,dot(100,um2px),dot(5,px2um)]))
# VideoSpermTracker_rev_26L60sec.m:798
        set(hRectangle,'FaceColor','w','EdgeColor','w')
        hText=text(440,460,'L. Urbano, et al (2015) Drexel University','FontSize',12)
# VideoSpermTracker_rev_26L60sec.m:802
        set(hText,'Color','w')
        pause(0.05)
        currFrame=getframe(gcf)
# VideoSpermTracker_rev_26L60sec.m:809
        writeVideo(vidObj,currFrame)
        if (k < numFrames):
            clf
        k
    # Close the movie file
    close_(vidObj)

# /////////////////////////////////////////////////////////////////////// #

#   Tracking Snapshot

# /////////////////////////////////////////////////////////////////////// #

snapShot=1
# VideoSpermTracker_rev_26L60sec.m:837
if (snapShot):
    figure
    set(gca,'Box','On')
    iptsetpref('ImshowBorder','tight')
    # Length of Trail History (in frames)
    trailLength=(dot(1,15))
# VideoSpermTracker_rev_26L60sec.m:845
    for k in 13 + concat([5,10,15,20,25,30]).reshape(-1):
        # Display the video frame
        imshow(imcomplement(rgb2gray(read(video,k))))
        hold('on')
        set(gcf,'Position',concat([255,90,955,715]))
        numTentativeTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 1),1)))
# VideoSpermTracker_rev_26L60sec.m:857
        numConfirmedTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2),1)))
# VideoSpermTracker_rev_26L60sec.m:860
        timeIdx=find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2)
# VideoSpermTracker_rev_26L60sec.m:865
        trackIdx=unique(TrackRecord(timeIdx,1))
# VideoSpermTracker_rev_26L60sec.m:869
        for trk in trackIdx.T.reshape(-1):
            # Get the indices to the data for this track up to time k
            dataIdx=find(TrackRecord(arange(),1) == logical_and(trk,TrackRecord(arange(),4)) <= k)
# VideoSpermTracker_rev_26L60sec.m:876
            SpMat[1,1]=TrackRecord(dataIdx(end()),22)
# VideoSpermTracker_rev_26L60sec.m:880
            SpMat[1,2]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:881
            SpMat[2,1]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:882
            SpMat[2,2]=TrackRecord(dataIdx(end()),24)
# VideoSpermTracker_rev_26L60sec.m:883
            if (length(dataIdx) <= trailLength):
                posX=multiply(TrackRecord(dataIdx,5),um2px)
# VideoSpermTracker_rev_26L60sec.m:888
                posY=multiply(TrackRecord(dataIdx,6),um2px)
# VideoSpermTracker_rev_26L60sec.m:889
                measX=multiply(TrackRecord(dataIdx,19),um2px)
# VideoSpermTracker_rev_26L60sec.m:890
                measY=multiply(TrackRecord(dataIdx,20),um2px)
# VideoSpermTracker_rev_26L60sec.m:891
            else:
                posX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),5),um2px)
# VideoSpermTracker_rev_26L60sec.m:893
                posY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),6),um2px)
# VideoSpermTracker_rev_26L60sec.m:894
                measX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),19),um2px)
# VideoSpermTracker_rev_26L60sec.m:895
                measY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),20),um2px)
# VideoSpermTracker_rev_26L60sec.m:896
            # Plot the Tracks and Measurements up to time k
            plot(posX,posY,'b')
            plot(measX,measY,'g')
            plot(measX,measY,'g.','MarkerSize',5)
            plot(measX(end()),measY(end()),'r+')
            # ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
        # set(ellipHand, 'Color', 'r');
            # Label the Track Number
            text(posX(end()) + 5,posY(end()) - 5,num2str(trk),'FontName','Arial','FontSize',12,'FontWeight','Bold','Color','g')
        # Zoom to frame
        set(gcf,'Position',concat([48,607,201,198]))
        # axis([330 330+150 25 25 + 150])
        axis(concat([80,80 + 150,220,220 + 150]))
        set(gcf,'PaperPositionMode','Auto')
        set(gcf,'renderer','painters')

# /////////////////////////////////////////////////////////////////////// #

#   Track Results

# /////////////////////////////////////////////////////////////////////// #
plotTrackResults=1
# VideoSpermTracker_rev_26L60sec.m:941
if (plotTrackResults):
    # Plot sperm tracks
    figure
    set(gcf,'Position',concat([0,276,1440,461]))
    subplot(2,4,concat([1,2,5,6]))
    hold('on')
    plotTrackHistory_2L(TrackRecord,T,um2px)
    set(gca,'Box','On','FontName','Arial','FontSize',11,'FontWeight','Bold')
    axis('equal')
    axis(concat([0,ROIx,0,ROIy]) / um2px)
    xlabel(concat(['X Position Coorindate (',char(181),'m)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    ylabel(concat(['Y Position Coorindate (',char(181),'m)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    title('Reconstructed Sperm Swimming Paths (Specimen A)','FontName','Arial','FontWeight','Bold','FontSize',12)

# # /////////////////////////////////////////////////////////////////////// #
# #
# #   2D-t plot
# #
# # /////////////////////////////////////////////////////////////////////// #
# plot2Dt = 1;
#
# if (plot2Dt)
#
#     figure; hold on; grid on; view(3);
#     plot2DtTrackHistory(TrackRecord, T, um2px);
#
# end

# /////////////////////////////////////////////////////////////////////// #

#   Sperm Motility Analysis

# /////////////////////////////////////////////////////////////////////// #
analyzeMotility=1
# VideoSpermTracker_rev_26L60sec.m:981
if (analyzeMotility):
    # [stats] = analyzeTrackRecord_rev_3L(TrackRecord, T);
# [stats] = analyzeTrackRecord_rev_4L(TrackRecord, T);
    stats=analyzeTrackRecord_rev_5L(TrackRecord,T)
# VideoSpermTracker_rev_26L60sec.m:987
    TRKNUM=stats.sampleTRK
# VideoSpermTracker_rev_26L60sec.m:989
    VCL=stats.sampleVCL
# VideoSpermTracker_rev_26L60sec.m:990
    VSL=stats.sampleVSL
# VideoSpermTracker_rev_26L60sec.m:991
    ALH=stats.sampleALH
# VideoSpermTracker_rev_26L60sec.m:992
    VAP=stats.sampleVAP
# VideoSpermTracker_rev_26L60sec.m:993
    LIN=stats.sampleLIN
# VideoSpermTracker_rev_26L60sec.m:994
    WOB=stats.sampleWOB
# VideoSpermTracker_rev_26L60sec.m:995
    STR=stats.sampleSTR
# VideoSpermTracker_rev_26L60sec.m:996
    MAD=stats.sampleMAD
# VideoSpermTracker_rev_26L60sec.m:997
    subplot(2,4,3)
    scatterXY(VSL,VCL,25,1)
    axis(concat([0,150,0,250]))
    grid('on')
    set(gca,'Box','On','FontName','Arial','FontSize',11,'FontWeight','Bold')
    xlabel(concat(['VSL (',char(181),'m/s)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    ylabel(concat(['VCL (',char(181),'m/s)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    # LIN vs ALH
    subplot(2,4,4)
    scatterXY(ALH,LIN,25,1)
    axis(concat([0,20,0,1]))
    grid('on')
    set(gca,'Box','On','FontName','Arial','FontSize',11,'FontWeight','Bold')
    xlabel(concat(['ALH (',char(181),'m)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    ylabel('LIN = VSL/VCL','FontName','Arial','FontSize',12,'FontWeight','Bold')
    # WOB vs VCL
    subplot(2,4,7)
    scatterXY(VSL,WOB,25,1)
    axis(concat([0,150,0,1]))
    grid('on')
    set(gca,'Box','On','FontName','Arial','FontSize',11,'FontWeight','Bold')
    xlabel(concat(['VSL (',char(181),'m/s)']),'FontName','Arial','FontSize',12,'FontWeight','Bold')
    ylabel('WOB = VAP/VCL','FontName','Arial','FontSize',12,'FontWeight','Bold')
    # LIN vs MAD
    subplot(2,4,8)
    scatterXY(MAD,LIN,25,1)
    axis(concat([1,180,0,1]))
    grid('on')
    set(gca,'Box','On','FontName','Arial','FontSize',11,'FontWeight','Bold')
    xlabel('MAD (deg)','FontName','Arial','FontSize',12,'FontWeight','Bold')
    ylabel('LIN = VSL/VCL','FontName','Arial','FontSize',12,'FontWeight','Bold')
    figuresize(20,6.5,'inches')
    set(gcf,'PaperPositionMode','Auto')
    set(gcf,'renderer','painters')
    disp(concat(['# Tracks Analyzed: ',num2str(stats.trkCount)]))
    disp(concat(['VCL: ',num2str(mean(stats.sampleVCL)),', ',num2str(std(stats.sampleVCL))]))
    disp(concat(['VSL: ',num2str(mean(stats.sampleVSL)),', ',num2str(std(stats.sampleVSL))]))
    disp(concat(['LIN: ',num2str(mean(stats.sampleLIN)),', ',num2str(std(stats.sampleLIN))]))
    disp(concat(['ALH: ',num2str(mean(stats.sampleALH)),', ',num2str(std(stats.sampleALH))]))
    disp(concat(['VAP: ',num2str(mean(stats.sampleVAP)),', ',num2str(std(stats.sampleVAP))]))
    disp(concat(['WOB: ',num2str(mean(stats.sampleWOB)),', ',num2str(std(stats.sampleWOB))]))
    disp(concat(['STR: ',num2str(mean(stats.sampleSTR)),', ',num2str(std(stats.sampleSTR))]))
    disp(concat(['MAD: ',num2str(mean(stats.sampleMAD)),', ',num2str(std(stats.sampleMAD))]))

# # /////////////////////////////////////////////////////////////////////// #
# #
# #   Analysis Time
# #
# # /////////////////////////////////////////////////////////////////////// #
# analysisTime = 1;

# if (analysisTime)

#     for jjj = 1:5

#         timePerSperm = jjj;

#         [stats] = analyzeTrackRecord_rev_6L(TrackRecord, T, timePerSperm);

#         meanVCL(jjj) = mean(stats.sampleVCL);
#         meanVSL(jjj) = mean(stats.sampleVSL);
#         meanALH(jjj) = mean(stats.sampleALH);
#         meanVAP(jjj) = mean(stats.sampleVAP);
#         meanLIN(jjj) = mean(stats.sampleLIN);
#         meanWOB(jjj) = mean(stats.sampleWOB);
#         meanSTR(jjj) = mean(stats.sampleSTR);
#         meanMAD(jjj) = mean(stats.sampleMAD);

#         stdVCL(jjj) = std(stats.sampleVCL);
#         stdVSL(jjj) = std(stats.sampleVSL);
#         stdALH(jjj) = std(stats.sampleALH);
#         stdVAP(jjj) = std(stats.sampleVAP);
#         stdLIN(jjj) = std(stats.sampleLIN);
#         stdWOB(jjj) = std(stats.sampleWOB);
#         stdSTR(jjj) = std(stats.sampleSTR);
#         stdMAD(jjj) = std(stats.sampleMAD);

#     end

#     figure; hold on; grid on;
#     plot(meanVCL, 'b')
#     plot(meanVCL+stdVCL(1:5), 'r--')
#     plot(meanVCL-stdVCL(1:5), 'r--')
#     axis([1 5 0 100])
# end

# # /////////////////////////////////////////////////////////////////////// #
# #
# #   Tracking Snapshot Sequence 1
# #
# # /////////////////////////////////////////////////////////////////////// #

# snapShot = 1;

# if (snapShot)

#     figure; set(gca, 'Box', 'On');
#     iptsetpref('ImshowBorder', 'tight')

#     # Length of Trail History (in frames)
#     trailLength = (1 * 15);

#     # Snapshot frame number
#     for k = ceil(42/T) + [1 2 3 4 5 6]*15

#         k

#         # Display the video frame
#         # imshow(imcomplement(rgb2gray(read(video, k))));
#         imshow(rgb2gray(read(video, k)));

#         # imshow(rgb2gray(read(video, k)));
#         hold on;
#         set(gcf, 'Position', [255 90 955 715]);

#         numTentativeTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));

#         numConfirmedTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));


#         # Find the Index to the data at time k for confirmed tracks
#         timeIdx = find(TrackRecord(:,4) == k & ...
#             TrackRecord(:,2) == 2);

#         # Find the tracks at time k
#         trackIdx = unique(TrackRecord(timeIdx,1));


#         # Plot each track up to time k
#         for trk = trackIdx'

#             # Get the indices to the data for this track up to time k
#             dataIdx = find(TrackRecord(:,1) == trk & ...
#                 TrackRecord(:,4) <= k);

#             # Predicted State, Covariance and Residual Covariance
#             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
#             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,2) = TrackRecord(dataIdx(end), 24);


#             # Plot the last trailLength# track points up to time k
#             if (length(dataIdx) <= trailLength)
#                 posX = TrackRecord(dataIdx,5) .* um2px;
#                 posY = TrackRecord(dataIdx,6) .* um2px;
#                 measX = TrackRecord(dataIdx,19) .* um2px;
#                 measY = TrackRecord(dataIdx,20) .* um2px;
#             else
#                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
#                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
#                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
#                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
#             end

#             # Plot the Tracks and Measurements up to time k
#             plot(measX, measY, 'g')
#             plot(measX, measY, 'g.', 'MarkerSize', 5);
#             plot(posX , posY, 'b');
#             plot(measX(end), measY(end), 'r+');

#             # Plot the track gate
#             # ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
#             # set(ellipHand, 'Color', 'c');

#             # Label the Track Number
#             text(posX(end)+5, posY(end)-5, num2str(trk), ...
#                 'FontName', 'Arial', 'FontSize', fontSize, 'FontWeight', 'Bold', 'Color', 'k');

#         end

#         # Zoom to frame
#         set(gcf, 'Position', [48   607   201   198])
#         #axis([330 330+150 25 25 + 150])
#         axis([360 360+150 120 120+150])

#         set(gcf, 'PaperPositionMode', 'Auto')
#         set(gcf, 'renderer', 'painters');
#         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_NewSnapshotFrame', num2str(k), '.png'])

#      end

# end



# # /////////////////////////////////////////////////////////////////////// #
# #
# #   Tracking Snapshot Sequence 2
# #
# # /////////////////////////////////////////////////////////////////////// #

# snapShot = 1;

# if (snapShot)

#     figure; set(gca, 'Box', 'On');
#     iptsetpref('ImshowBorder', 'tight')

#     # Length of Trail History (in frames)
#     trailLength = (1 * 15);

#     # Snapshot frame number
#     for k = ceil(35/T) + [1 2 3 4 5 6]*7

#         k

#         # Display the video frame
#         # imshow(imcomplement(rgb2gray(read(video, k))));
#         imshow(rgb2gray(read(video, k)));

#         # imshow(rgb2gray(read(video, k)));
#         hold on;
#         set(gcf, 'Position', [255 90 955 715]);

#         numTentativeTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));

#         numConfirmedTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));


#         # Find the Index to the data at time k for confirmed tracks
#         timeIdx = find(TrackRecord(:,4) == k & ...
#             TrackRecord(:,2) == 2);

#         # Find the tracks at time k
#         trackIdx = unique(TrackRecord(timeIdx,1));


#         # Plot each track up to time k
#         for trk = trackIdx'

#             # Get the indices to the data for this track up to time k
#             dataIdx = find(TrackRecord(:,1) == trk & ...
#                 TrackRecord(:,4) <= k);

#             # Predicted State, Covariance and Residual Covariance
#             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
#             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,2) = TrackRecord(dataIdx(end), 24);


#             # Plot the last trailLength# track points up to time k
#             if (length(dataIdx) <= trailLength)
#                 posX = TrackRecord(dataIdx,5) .* um2px;
#                 posY = TrackRecord(dataIdx,6) .* um2px;
#                 measX = TrackRecord(dataIdx,19) .* um2px;
#                 measY = TrackRecord(dataIdx,20) .* um2px;
#             else
#                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
#                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
#                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
#                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
#             end

#             # Plot the Tracks and Measurements up to time k
#             plot(measX, measY, 'g')
#             plot(measX, measY, 'g.', 'MarkerSize', 5);
#             plot(posX , posY, 'b');
#             plot(measX(end), measY(end), 'r+');

#             # Plot the track gate
#             # ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
#             # set(ellipHand, 'Color', 'c');

#             # Label the Track Number
#             text(posX(end)+5, posY(end)-5, num2str(trk), ...
#                 'FontName', 'Arial', 'FontSize', fontSize, 'FontWeight', 'Bold', 'Color', 'k');

#         end

#         # Zoom to frame
#         set(gcf, 'Position', [48   607   201   198])
#         # axis([330 330+150 25 25 + 150])
#         # axis([360 360+150 120 120+150])
#         axis([180 180+150 108 108+150]);#

#         set(gcf, 'PaperPositionMode', 'Auto')
#         set(gcf, 'renderer', 'painters');
#         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_Seq2SnapshotFrame', num2str(k), '.png'])

#      end

# end




# # /////////////////////////////////////////////////////////////////////// #
# #
# #   Tracking Snapshot Sequence 3
# #
# # /////////////////////////////////////////////////////////////////////// #

# snapShot = 1;

# if (snapShot)

#     figure; set(gca, 'Box', 'On');
#     iptsetpref('ImshowBorder', 'tight')

#     # Length of Trail History (in frames)
#     trailLength = (1 * 15);

#     # Snapshot frame number
#     for k = ceil(13/T) + [1 2 3 4 5 6]*7

#         k

#         # Display the video frame
#         # imshow(imcomplement(rgb2gray(read(video, k))));
#         imshow(rgb2gray(read(video, k)));

#         # imshow(rgb2gray(read(video, k)));
#         hold on;
#         set(gcf, 'Position', [255 90 955 715]);

#         numTentativeTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));

#         numConfirmedTracks(k) = length(unique(TrackRecord(...
#             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));


#         # Find the Index to the data at time k for confirmed tracks
#         timeIdx = find(TrackRecord(:,4) == k & ...
#             TrackRecord(:,2) == 2);

#         # Find the tracks at time k
#         trackIdx = unique(TrackRecord(timeIdx,1));


#         # Plot each track up to time k
#         for trk = trackIdx'

#             # Get the indices to the data for this track up to time k
#             dataIdx = find(TrackRecord(:,1) == trk & ...
#                 TrackRecord(:,4) <= k);

#             # Predicted State, Covariance and Residual Covariance
#             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
#             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
#             SpMat(2,2) = TrackRecord(dataIdx(end), 24);


#             # Plot the last trailLength# track points up to time k
#             if (length(dataIdx) <= trailLength)
#                 posX = TrackRecord(dataIdx,5) .* um2px;
#                 posY = TrackRecord(dataIdx,6) .* um2px;
#                 measX = TrackRecord(dataIdx,19) .* um2px;
#                 measY = TrackRecord(dataIdx,20) .* um2px;
#             else
#                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
#                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
#                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
#                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
#             end

#             # Plot the Tracks and Measurements up to time k
#             plot(measX, measY, 'g')
#             plot(measX, measY, 'g.', 'MarkerSize', 5);
#             plot(posX , posY, 'b');
#             plot(measX(end), measY(end), 'r+');

#             # Plot the track gate
#             # ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
#             # set(ellipHand, 'Color', 'c');

#             # Label the Track Number
#             text(posX(end)+5, posY(end)-5, num2str(trk), ...
#                 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold', 'Color', 'k');

#         end

#         # Zoom to frame
#         set(gcf, 'Position', [48   607   201   198])
#         # axis([330 330+150 25 25 + 150])
#         # axis([360 360+150 120 120+150])
#         # axis([180 180+150 108 108+150]);#
#         # axis([25 25+150 275 275+150]);
#         axis([380 380+150 260 260+150]);

#         set(gcf, 'PaperPositionMode', 'Auto')
#         set(gcf, 'renderer', 'painters');
#         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_Seq3SnapshotFrame', num2str(k), '.png'])

#      end

# end

# /////////////////////////////////////////////////////////////////////// #

#   Tracking Snapshot Sequence 4

# /////////////////////////////////////////////////////////////////////// #

snapShot=0
# VideoSpermTracker_rev_26L60sec.m:1426
if (snapShot):
    figure
    set(gca,'Box','On')
    iptsetpref('ImshowBorder','tight')
    # Length of Trail History (in frames)
    trailLength=(dot(1,15))
# VideoSpermTracker_rev_26L60sec.m:1434
    for k in ceil(29 / T) + dot(concat([1,2,3,4,5,6]),5).reshape(-1):
        k
        # Display the video frame
    # imshow(imcomplement(rgb2gray(read(video, k))));
        imshow(rgb2gray(read(video,k)))
        hold('on')
        set(gcf,'Position',concat([255,90,955,715]))
        numTentativeTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 1),1)))
# VideoSpermTracker_rev_26L60sec.m:1449
        numConfirmedTracks[k]=length(unique(TrackRecord(find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2),1)))
# VideoSpermTracker_rev_26L60sec.m:1452
        timeIdx=find(TrackRecord(arange(),4) == logical_and(k,TrackRecord(arange(),2)) == 2)
# VideoSpermTracker_rev_26L60sec.m:1457
        trackIdx=unique(TrackRecord(timeIdx,1))
# VideoSpermTracker_rev_26L60sec.m:1461
        for trk in trackIdx.T.reshape(-1):
            # Get the indices to the data for this track up to time k
            dataIdx=find(TrackRecord(arange(),1) == logical_and(trk,TrackRecord(arange(),4)) <= k)
# VideoSpermTracker_rev_26L60sec.m:1468
            SpMat[1,1]=TrackRecord(dataIdx(end()),22)
# VideoSpermTracker_rev_26L60sec.m:1472
            SpMat[1,2]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:1473
            SpMat[2,1]=TrackRecord(dataIdx(end()),23)
# VideoSpermTracker_rev_26L60sec.m:1474
            SpMat[2,2]=TrackRecord(dataIdx(end()),24)
# VideoSpermTracker_rev_26L60sec.m:1475
            if (length(dataIdx) <= trailLength):
                posX=multiply(TrackRecord(dataIdx,5),um2px)
# VideoSpermTracker_rev_26L60sec.m:1480
                posY=multiply(TrackRecord(dataIdx,6),um2px)
# VideoSpermTracker_rev_26L60sec.m:1481
                measX=multiply(TrackRecord(dataIdx,19),um2px)
# VideoSpermTracker_rev_26L60sec.m:1482
                measY=multiply(TrackRecord(dataIdx,20),um2px)
# VideoSpermTracker_rev_26L60sec.m:1483
            else:
                posX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),5),um2px)
# VideoSpermTracker_rev_26L60sec.m:1485
                posY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),6),um2px)
# VideoSpermTracker_rev_26L60sec.m:1486
                measX=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),19),um2px)
# VideoSpermTracker_rev_26L60sec.m:1487
                measY=multiply(TrackRecord(dataIdx(arange(end() - trailLength,end(),1)),20),um2px)
# VideoSpermTracker_rev_26L60sec.m:1488
            # Plot the Tracks and Measurements up to time k
            plot(measX,measY,'g')
            plot(measX,measY,'g.','MarkerSize',5)
            plot(posX,posY,'b')
            plot(measX(end()),measY(end()),'r+')
            # ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
        # set(ellipHand, 'Color', 'c');
            # Label the Track Number
            text(posX(end()) + 5,posY(end()) - 5,num2str(trk),'FontName','Arial','FontSize',12,'FontWeight','Bold','Color','k')
        # Zoom to frame
        set(gcf,'Position',concat([48,607,201,198]))
        # axis([330 330+150 25 25 + 150])
    # axis([360 360+150 120 120+150])
    # axis([180 180+150 108 108+150]);#
    # axis([25 25+150 275 275+150]);
    # axis([380 380+150 260 260+150]);
    # axis([400 400+150 100 100+150]);
        axis(concat([280,280 + 150,220,220 + 150]))
        set(gcf,'PaperPositionMode','Auto')
        set(gcf,'renderer','painters')
        print_(gcf,'-dpng','-r300',concat([videoFile,'MTT',num2str(mttAlgorithm),'_Seq4SnapshotFrame',num2str(k),'.png']))
