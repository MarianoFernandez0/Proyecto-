# Generated with SMOP  0.41
from libsmop import *
# nnAssociation.m

    
@function
def nnAssociation(LR=None,*args,**kwargs):
    varargin = nnAssociation.varargin
    nargin = nnAssociation.nargin

    # Size of the problem matrix
    m,n=size(LR,nargout=2)
# nnAssociation.m:4
    # Initialize beta matrix to zeros
    beta=zeros(m,n)
# nnAssociation.m:7
    # Negative Log-likelihood Ratio
    NLLR=- log(LR)
# nnAssociation.m:10
    # Calculate beta_jt
    for t in arange(1,n).reshape(-1):
        # Nearest Neighbor (measurement with maximum likelihood)
        j=find(NLLR(arange(),t) == min(NLLR(arange(),t)))
# nnAssociation.m:16
        if length(j) > 1:
            beta[j,t]=0
# nnAssociation.m:20
        else:
            beta[j,t]=1
# nnAssociation.m:22
    