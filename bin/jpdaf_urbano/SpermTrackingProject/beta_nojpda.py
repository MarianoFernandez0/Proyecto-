# Generated with SMOP  0.41
from libsmop import *
# beta_nojpda.m

    
@function
def beta_nojpda(L=None,PD=None,PG=None,*args,**kwargs):
    varargin = beta_nojpda.varargin
    nargin = beta_nojpda.nargin

    #   
#   Nearly-optimal JPDA association probability calculator uses Murty's
#   N-best ranked assignments to find the 10 most likely joint events
#   
#   Usage:  
#   
#       beta(j,t) = beta_nojpda(L, PD, PG)
    
    # ####################################################################### #
    
    #   Calculate negative log likelihood ratio (NLLR) cost matrix A
    
    # ####################################################################### #
    
    # Negative log likelihood ratio matrix
    A_orig=- log(L)
# beta_nojpda.m:19
    # Number of measurements Nm, number of tracks Nt
    m,n=size(A_orig,nargout=2)
# beta_nojpda.m:22
    # ####################################################################### #
    
    #   Find N-best assignments using Murty's method
    
    # ####################################################################### #
    
    # Initial problem
    P0=copy(A_orig)
# beta_nojpda.m:33
    # Initial (0-th) optimal assignment
    S0,V0=munkres(P0,nargout=2)
# beta_nojpda.m:36
    # Calculate the N-best solutions - here, N = 25
    SOLUTIONS=murtys_best(P0,S0,V0,100)
# beta_nojpda.m:39
    # Number of events found
    num_events=length(SOLUTIONS)
# beta_nojpda.m:42
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    for N in arange(1,num_events).reshape(-1):
        # Hadamard product of L and the N-th best ranked assignment
        THETA=multiply(L,SOLUTIONS[N])
# beta_nojpda.m:54
        tau_j,delta_t,L_jt=find(THETA,nargout=3)
# beta_nojpda.m:57
        term1=prod(L_jt)
# beta_nojpda.m:60
        term2=dot((PD) ** (length(tau_j)),(1 - PD) ** (n - length(delta_t)))
# beta_nojpda.m:63
        P_THETA[N]=dot(term1,term2)
# beta_nojpda.m:66
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# beta_nojpda.m:71
    # Normalize the probabilities
    for N in arange(1,num_events).reshape(-1):
        P_THETA[N]=P_THETA[N] / c
# beta_nojpda.m:76
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # Initialize the beta matrix
    beta=zeros(m,n)
# beta_nojpda.m:89
    # For each target
    for t in arange(1,n).reshape(-1):
        # For each measurement
        for j in arange(1,m).reshape(-1):
            # For each association event
            for N in arange(1,num_events).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],SOLUTIONS[N](j,t))
# beta_nojpda.m:100
    