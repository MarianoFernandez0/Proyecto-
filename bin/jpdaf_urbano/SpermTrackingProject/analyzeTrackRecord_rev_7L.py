# Generated with SMOP  0.41
from libsmop import *
# analyzeTrackRecord_rev_7L.m

    
@function
def analyzeTrackRecord(TrackRecord=None,T=None,timePerSperm=None,*args,**kwargs):
    varargin = analyzeTrackRecord.varargin
    nargin = analyzeTrackRecord.nargin

    # /////////////////////////////////////////////////////////////////////////
    
    #   Measure Sperm Motility Parameters from Track File
#   by Leonardo F. Urbano
#   April 5th, 2015
    
    # /////////////////////////////////////////////////////////////////////////
    
    # Time avg window (s)
    avgTime=1
# analyzeTrackRecord_rev_7L.m:12
    # Number of points in avg window
    windowSize=ceil(avgTime / T)
# analyzeTrackRecord_rev_7L.m:15
    # Analysis time per sperm  (sec)
# timePerSperm = 1;
    
    # Clear the Records
    sampleTRK=[]
# analyzeTrackRecord_rev_7L.m:21
    sampleVCL=[]
# analyzeTrackRecord_rev_7L.m:22
    sampleVSL=[]
# analyzeTrackRecord_rev_7L.m:23
    sampleALH=[]
# analyzeTrackRecord_rev_7L.m:24
    sampleLIN=[]
# analyzeTrackRecord_rev_7L.m:25
    sampleVAP=[]
# analyzeTrackRecord_rev_7L.m:26
    sampleWOB=[]
# analyzeTrackRecord_rev_7L.m:27
    sampleSTR=[]
# analyzeTrackRecord_rev_7L.m:28
    sampleMAD=[]
# analyzeTrackRecord_rev_7L.m:29
    # Number of Confirmed Tracks to Analyze
    trackList=unique(TrackRecord((TrackRecord(arange(),2) == 2),1))
# analyzeTrackRecord_rev_7L.m:32
    numTracks=length(trackList)
# analyzeTrackRecord_rev_7L.m:33
    # Draw wait bar
    hWaitbar=waitbar(0,'Sperm analysis in progress... ')
# analyzeTrackRecord_rev_7L.m:36
    # Initialize the track count for the waitbar
    trkCount=0
# analyzeTrackRecord_rev_7L.m:39
    # Minimum number of track points to be analyzed
    numPoints=ceil(dot(timePerSperm,windowSize))
# analyzeTrackRecord_rev_7L.m:42
    # Analyze each track
    for trk in trackList.T.reshape(-1):
        # Set of measurements for this track
        dataIdx=find(TrackRecord(arange(),1) == trk)
# analyzeTrackRecord_rev_7L.m:48
        dataIdx=dataIdx(arange(5,end() - 5))
# analyzeTrackRecord_rev_7L.m:51
        # motility analysis
        if (length(dataIdx) > numPoints):
            # Take only numPoints worth of data
            dataIdx=dataIdx(arange(1,numPoints))
# analyzeTrackRecord_rev_7L.m:58
            trkCount=trkCount + 1
# analyzeTrackRecord_rev_7L.m:61
            measX=TrackRecord(dataIdx,19)
# analyzeTrackRecord_rev_7L.m:64
            measY=TrackRecord(dataIdx,20)
# analyzeTrackRecord_rev_7L.m:65
            for kStep in arange(0,(numPoints - windowSize)).reshape(-1):
                VCL=[]
# analyzeTrackRecord_rev_7L.m:70
                VSL=[]
# analyzeTrackRecord_rev_7L.m:70
                LIN=[]
# analyzeTrackRecord_rev_7L.m:70
                ALH=[]
# analyzeTrackRecord_rev_7L.m:70
                VAP=[]
# analyzeTrackRecord_rev_7L.m:71
                WOB=[]
# analyzeTrackRecord_rev_7L.m:71
                MAD=[]
# analyzeTrackRecord_rev_7L.m:71
                STR=[]
# analyzeTrackRecord_rev_7L.m:71
                Zx=measX(arange(kStep + 1,kStep + windowSize)).T
# analyzeTrackRecord_rev_7L.m:74
                Zy=measY(arange(kStep + 1,kStep + windowSize)).T
# analyzeTrackRecord_rev_7L.m:75
                Vx=diff(Zx) / T
# analyzeTrackRecord_rev_7L.m:78
                Vy=diff(Zy) / T
# analyzeTrackRecord_rev_7L.m:79
                VCL=mean(sqrt(Vx ** 2 + Vy ** 2))
# analyzeTrackRecord_rev_7L.m:80
                DSLx=Zx(end()) - Zx(1)
# analyzeTrackRecord_rev_7L.m:83
                DSLy=Zy(end()) - Zy(1)
# analyzeTrackRecord_rev_7L.m:84
                VSL=sqrt(DSLx ** 2 + DSLy ** 2) / (dot(windowSize,T))
# analyzeTrackRecord_rev_7L.m:85
                LIN=VSL / VCL
# analyzeTrackRecord_rev_7L.m:88
                Sx=[]
# analyzeTrackRecord_rev_7L.m:91
                Sy=[]
# analyzeTrackRecord_rev_7L.m:91
                for jjj in arange(3,(length(Zx) - 2)).reshape(-1):
                    Sx=concat([Sx,mean(Zx(arange(jjj - 2,jjj + 2)))])
# analyzeTrackRecord_rev_7L.m:93
                    Sy=concat([Sy,mean(Zy(arange(jjj - 2,jjj + 2)))])
# analyzeTrackRecord_rev_7L.m:94
                # Velocity average path
                VSx=diff(Sx) / T
# analyzeTrackRecord_rev_7L.m:98
                VSy=diff(Sy) / T
# analyzeTrackRecord_rev_7L.m:99
                VAP=mean(sqrt(VSx ** 2 + VSy ** 2))
# analyzeTrackRecord_rev_7L.m:100
                DLH=concat([[Zx(arange(3,end() - 2))],[Zy(arange(3,end() - 2))]]) - concat([[Sx],[Sy]])
# analyzeTrackRecord_rev_7L.m:103
                DEV=sqrt(DLH(1,arange()) ** 2 + DLH(2,arange()) ** 2)
# analyzeTrackRecord_rev_7L.m:104
                ALH=dot(2,mean(DEV))
# analyzeTrackRecord_rev_7L.m:105
                # ALH = 2 * max(DEV);
                # Mean Angular Displacement
                MADi=[]
# analyzeTrackRecord_rev_7L.m:109
                for jjj in arange(2,length(Vx)).reshape(-1):
                    mag1=norm(concat([Vx(jjj),Vy(jjj)]))
# analyzeTrackRecord_rev_7L.m:111
                    uv1=concat([[Vx(jjj)],[Vy(jjj)]]) / mag1
# analyzeTrackRecord_rev_7L.m:112
                    mag2=norm(concat([Vx(jjj - 1),Vy(jjj - 1)]))
# analyzeTrackRecord_rev_7L.m:113
                    uv2=concat([[Vx(jjj - 1)],[Vy(jjj - 1)]]) / mag2
# analyzeTrackRecord_rev_7L.m:114
                    MADi=concat([MADi,acosd(dot(uv1,uv2))])
# analyzeTrackRecord_rev_7L.m:115
                MAD=mean(MADi)
# analyzeTrackRecord_rev_7L.m:117
                if isreal(MAD):
                    MAD=copy(MAD)
# analyzeTrackRecord_rev_7L.m:119
                else:
                    MAD=0
# analyzeTrackRecord_rev_7L.m:121
                # Wobble
                WOB=VAP / VCL
# analyzeTrackRecord_rev_7L.m:126
                STR=VSL / VAP
# analyzeTrackRecord_rev_7L.m:127
                if (VCL > 30) and (VCL <= 250) and (VSL > 0) and (VSL <= 150):
                    sampleTRK=concat([sampleTRK,trk])
# analyzeTrackRecord_rev_7L.m:132
                    sampleVCL=concat([sampleVCL,VCL])
# analyzeTrackRecord_rev_7L.m:133
                    sampleVSL=concat([sampleVSL,VSL])
# analyzeTrackRecord_rev_7L.m:134
                    sampleLIN=concat([sampleLIN,LIN])
# analyzeTrackRecord_rev_7L.m:135
                    sampleALH=concat([sampleALH,ALH])
# analyzeTrackRecord_rev_7L.m:136
                    sampleVAP=concat([sampleVAP,VAP])
# analyzeTrackRecord_rev_7L.m:137
                    sampleWOB=concat([sampleWOB,WOB])
# analyzeTrackRecord_rev_7L.m:138
                    sampleSTR=concat([sampleSTR,STR])
# analyzeTrackRecord_rev_7L.m:139
                    sampleMAD=concat([sampleMAD,MAD])
# analyzeTrackRecord_rev_7L.m:140
        # Update the waitbar
        waitbar(trkCount / numTracks,hWaitbar)
    
    # Store stats
    stats.sampleTRK = copy(sampleTRK)
# analyzeTrackRecord_rev_7L.m:154
    stats.sampleVCL = copy(sampleVCL)
# analyzeTrackRecord_rev_7L.m:155
    stats.sampleVSL = copy(sampleVSL)
# analyzeTrackRecord_rev_7L.m:156
    stats.sampleLIN = copy(sampleLIN)
# analyzeTrackRecord_rev_7L.m:157
    stats.sampleALH = copy(sampleALH)
# analyzeTrackRecord_rev_7L.m:158
    stats.sampleVAP = copy(sampleVAP)
# analyzeTrackRecord_rev_7L.m:159
    stats.sampleWOB = copy(sampleWOB)
# analyzeTrackRecord_rev_7L.m:160
    stats.sampleSTR = copy(sampleSTR)
# analyzeTrackRecord_rev_7L.m:161
    stats.sampleMAD = copy(sampleMAD)
# analyzeTrackRecord_rev_7L.m:162
    stats.trackCount = copy(trkCount)
# analyzeTrackRecord_rev_7L.m:163
    close_(hWaitbar)