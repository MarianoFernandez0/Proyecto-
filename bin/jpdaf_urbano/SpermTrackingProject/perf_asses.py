# Generated with SMOP  0.41
from libsmop import *
# perf_asses.m

    
@function
def perf_asses(trk=None,est_trk=None,apar=None,*args,**kwargs):
    varargin = perf_asses.varargin
    nargin = perf_asses.nargin

    # trk - true tracks3
# est_trk - estimated tracks
# VER 2: : new assignment of estimated to true tracks
    
    if nargin == 3:
        OSPA.l = copy(apar)
# perf_asses.m:8
    else:
        OSPA.l = copy(25)
# perf_asses.m:10
    
    PLOT=1
# perf_asses.m:14
    OSPA.p = copy(1)
# perf_asses.m:16
    OSPA.c = copy(25)
# perf_asses.m:17
    a1,K1,num_trk=size(trk,nargout=3)
# perf_asses.m:20
    a2,K2,num_etrk=size(est_trk,nargout=3)
# perf_asses.m:21
    if logical_or((a1 != a2),(K1 != K2)):
        error('wrong input')
    
    # find the global best assignment of true tracks to estimated tracks
    DELTA=80
# perf_asses.m:28
    D=zeros(num_trk,num_etrk)
# perf_asses.m:29
    for i in arange(1,num_trk).reshape(-1):
        t=trk(arange(),arange(),i)
# perf_asses.m:31
        for j in arange(1,num_etrk).reshape(-1):
            et=est_trk(arange(),arange(),j)
# perf_asses.m:33
            cnt=0
# perf_asses.m:34
            for k in arange(1,K1).reshape(-1):
                if logical_or(not_(isnan(t(1,k))),not_(isnan(et(1,k)))):
                    d=sqrt(sum((t(arange(),k) - et(arange(),k)) ** 2))
# perf_asses.m:37
                    if not_(isnan(d)):
                        D[i,j]=D(i,j) + min(DELTA,d)
# perf_asses.m:39
                    else:
                        D[i,j]=D(i,j) + DELTA
# perf_asses.m:41
            D[i,j]=D(i,j) / K1
# perf_asses.m:45
    
    Matching,Cost=Hungarian(D,nargout=2)
# perf_asses.m:51
    for i in arange(1,num_trk).reshape(-1):
        trk_corr[i]=find(Matching(i,arange()) == 1)
# perf_asses.m:53
    
    # plot
    if PLOT:
        v=concat([0.1,0.5,0.9])
# perf_asses.m:59
        ix=0
# perf_asses.m:60
        for i in arange(1,length(v)).reshape(-1):
            for j in arange(1,length(v)).reshape(-1):
                for k in arange(1,length(v)).reshape(-1):
                    ix=ix + 1
# perf_asses.m:64
                    col[ix,arange()]=concat([v(i),v(j),v(k)])
# perf_asses.m:65
        figure(10)
        for i in arange(1,num_trk).reshape(-1):
            plot(trk(1,arange(),i),trk(3,arange(),i),'-','Color',col(dot(2,i),arange()))
            hold('on')
            plot(est_trk(1,arange(),trk_corr(i)),est_trk(3,arange(),trk_corr(i)),':','Color',col(dot(2,i),arange()))
        hold('off')
    
    # For every k assign labels to estimated tracks and runs OSPA
    for k in arange(1,K1).reshape(-1):
        X=[]
# perf_asses.m:80
        Xl=[]
# perf_asses.m:81
        for i in arange(1,num_trk).reshape(-1):
            if not_(isnan(trk(1,k,i))):
                X=concat([X,trk(arange(),k,i)])
# perf_asses.m:84
                Xl=concat([Xl,i])
# perf_asses.m:85
        card_X[k]=size(X,2)
# perf_asses.m:88
        Y=[]
# perf_asses.m:89
        Yl=[]
# perf_asses.m:90
        for i in arange(1,num_etrk).reshape(-1):
            if not_(isnan(est_trk(1,k,i))):
                Y=concat([Y,est_trk(arange(),k,i)])
# perf_asses.m:93
                ix=find(trk_corr == i)
# perf_asses.m:94
                if not_(isempty(ix)):
                    Yl=concat([Yl,ix])
# perf_asses.m:96
                else:
                    Yl=concat([Yl,12345])
# perf_asses.m:98
        card_Y[k]=size(Y,2)
# perf_asses.m:102
        dist(k),loce(k),carde(k)=trk_ospa_dist(X,Xl,Y,Yl,OSPA,nargout=3)
# perf_asses.m:104
    
    if PLOT:
        figure(2)
        plot(concat([arange(1,K1)]),card_X,'r')
        hold('on')
        plot(concat([arange(1,K1)]),card_Y,'b')
        hold('off')
        figure(3)
        plot(concat([arange(1,K1)]),dist,'r','Linewidth',2)
        hold('on')
        plot(concat([arange(1,K1)]),loce,'b--',concat([arange(1,K1)]),carde,'k:')
        hold('off')
    