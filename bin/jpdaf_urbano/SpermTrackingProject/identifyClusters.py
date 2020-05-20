# Generated with SMOP  0.41
from libsmop import *
# identifyClusters.m

    
@function
def identifyClusters(A=None,*args,**kwargs):
    varargin = identifyClusters.varargin
    nargin = identifyClusters.nargin

    
    #   Identifies Track Clusters
    
    #   By Leonardo F. Urbano
    
    #   23 September 2014
    
    # clear all
# close all
    
    # THREE CLUSTERS
# A = [0 1 1 0 1 0 0;
#      1 0 0 0 0 1 0;
#      0 0 0 1 0 0 1;
#      0 0 0 0 0 1 0;
#      0 0 0 1 0 0 0;
#      0 0 1 0 1 0 0];
    
    # ONE CLUSTER
# A = [0 1 0 1 1 0 1;
#      1 0 1 0 0 1 0;
#      0 0 0 1 0 0 0;
#      0 1 0 0 1 0 1;
#      1 1 0 0 0 1 0];
    
    # Possible Outcomes
# if A is empty : there are no measurements at all 
#   every track gets its own cluster
    
    mA,nA=size(A,nargout=2)
# identifyClusters.m:36
    i,j=find(A,nargout=2)
# identifyClusters.m:37
    if (mA == 1):
        C=concat([zeros(length(i),1),i.T,j.T])
# identifyClusters.m:39
    else:
        C=concat([zeros(length(i),1),i,j])
# identifyClusters.m:41
    
    C[1,1]=1
# identifyClusters.m:43
    # // Initialize
    output=[]
# identifyClusters.m:47
    clusterNumber=1
# identifyClusters.m:48
    listDiff=1
# identifyClusters.m:49
    while (sum(C(arange(),2)) > 0) and (sum(C(arange(),3)) > 0):

        while (listDiff > 0):

            list1=find(C(arange(),1)).T
# identifyClusters.m:54
            for m in list1.reshape(-1):
                i=C(m,2)
# identifyClusters.m:58
                foundRows=find(C(arange(),2) == i)
# identifyClusters.m:59
                C[foundRows,1]=1
# identifyClusters.m:60
                list2=find(C(arange(),1)).T
# identifyClusters.m:62
                for n in list2.reshape(-1):
                    j=C(n,3)
# identifyClusters.m:66
                    foundCols=find(C(arange(),3) == j)
# identifyClusters.m:67
                    C[foundCols,1]=1
# identifyClusters.m:68
            list3=find(C(arange(),1)).T
# identifyClusters.m:74
            listDiff=length(list3) - length(list1)
# identifyClusters.m:75

        # Write Output
        D=C(list3,arange())
# identifyClusters.m:80
        D[arange(),1]=clusterNumber
# identifyClusters.m:81
        output=concat([[output],[D]])
# identifyClusters.m:82
        clusterNumber=clusterNumber + 1
# identifyClusters.m:83
        C[list3,arange()]=0
# identifyClusters.m:86
        C[find(C(arange(),2) > 0,1,'First'),1]=1
# identifyClusters.m:89
        listDiff=1
# identifyClusters.m:92

    
    # If there are no clusters found then give each track its own cluster
    if isempty(output):
        for ttt in arange(1,nA).reshape(-1):
            output=concat([[output],[ttt,0,ttt]])
# identifyClusters.m:101
    else:
        # If at least one cluster is found, then give the remaining tracks
    # their own clusters
        tNoMeas=setdiff(arange(1,nA),output(arange(),3))
# identifyClusters.m:107
        if logical_not(isempty(tNoMeas)):
            for ttt in tNoMeas.reshape(-1):
                output=concat([[output],[concat([max(output(arange(),1)) + 1,0,ttt])]])
# identifyClusters.m:111
    
    # output
    