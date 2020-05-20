# Generated with SMOP  0.41
from libsmop import *
# beta_nnjpda.m

    
@function
def beta_nnjpda(L=None,PD=None,PG=None,*args,**kwargs):
    varargin = beta_nnjpda.varargin
    nargin = beta_nnjpda.nargin

    #   
#   Nearest-neighbor JPDA association probability calculator uses
#   Fitzgerald's method for avoiding track coaelescence
#   
#   Usage:  
#   
#       beta(j,t) = beta_nnjpda(L, PD, PG)
    
    # ####################################################################### #
    
    #   Calculate negative log likelihood ratio (NLLR) cost matrix A
    
    # ####################################################################### #
    
    # Negative log likelihood ratio matrix
    A_orig=- log(L)
# beta_nnjpda.m:19
    # Number of measurements Nm, number of tracks Nt
    m,n=size(A_orig,nargout=2)
# beta_nnjpda.m:22
    # ####################################################################### #
    
    #   Find N-best assignments using Murty's method
    
    # ####################################################################### #
    
    # Initial problem
    P0=copy(A_orig)
# beta_nnjpda.m:33
    # Initial (0-th) optimal assignment
    S0,V0=munkres(P0,nargout=2)
# beta_nnjpda.m:36
    # Calculate the N-best solutions - here, N = 10
    SOLUTIONS=murtys_best(P0,S0,V0,10)
# beta_nnjpda.m:39
    # Number of events found
    num_events=length(SOLUTIONS)
# beta_nnjpda.m:42
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    for N in arange(1,num_events).reshape(-1):
        # Hadamard product of L and the N-th best ranked assignment
        THETA=multiply(L,SOLUTIONS[N])
# beta_nnjpda.m:55
        tau_j,delta_t,L_jt=find(THETA,nargout=3)
# beta_nnjpda.m:58
        term1=prod(L_jt)
# beta_nnjpda.m:61
        term2=dot((PD) ** (length(tau_j)),(1 - PD) ** (n - length(delta_t)))
# beta_nnjpda.m:64
        P_THETA[N]=dot(term1,term2)
# beta_nnjpda.m:67
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# beta_nnjpda.m:72
    # Normalize the probabilities
    for N in arange(1,num_events).reshape(-1):
        P_THETA[N]=P_THETA[N] / c
# beta_nnjpda.m:77
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # Initialize the beta matrix
    beta=zeros(m,n)
# beta_nnjpda.m:90
    # For each target
    for t in arange(1,n).reshape(-1):
        # For each measurement
        for j in arange(1,m).reshape(-1):
            # For each association event
            for N in arange(1,num_events).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],SOLUTIONS[N](j,t))
# beta_nnjpda.m:101
    
    # ########################################################### #
    
    #   Choose the best assignments from the probabilities
    
    # ########################################################### #
    
    # Minimize the sum of 1 - probabilities
    beta=munkres(1 - beta)
# beta_nnjpda.m:119