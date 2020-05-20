# Generated with SMOP  0.41
from libsmop import *
# greedy.m

    
@function
def greedy(A=None,*args,**kwargs):
    varargin = greedy.varargin
    nargin = greedy.nargin

    # clear all 
# close all
# 
# A = randn(5,5)
    
    # Size of A
    m,n=size(A,nargout=2)
# greedy.m:9
    # Assignment matrix
    A_opt=zeros(m,n)
# greedy.m:12
    # Column index
    cols=arange(1,n)
# greedy.m:15
    for j in arange(1,m).reshape(-1):
        if (min(A(j,cols)) != inf):
            __,t=find((A == min(A(j,cols))),nargout=2)
# greedy.m:21
            A_opt[j,t]=1
# greedy.m:23
            cols=cols(cols != t)
# greedy.m:25
    
    