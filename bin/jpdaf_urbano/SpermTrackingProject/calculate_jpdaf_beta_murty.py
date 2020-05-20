# Generated with SMOP  0.41
from libsmop import *
# calculate_jpdaf_beta_murty.m

    
@function
def calculate_jpdaf_beta_murty(A=None,PD=None,*args,**kwargs):
    varargin = calculate_jpdaf_beta_murty.varargin
    nargin = calculate_jpdaf_beta_murty.nargin

    #   Sub-optimal Joint Probabilistic Data Association (JPDA)
#   Joint Association Probability Calculator using murty's k-best
#   
#   Usage:  
#   
#       beta(j,t) = calculate_jpdaf_beta_murty(f, A, PD)
    
    #
    
    # Convert from negative log likelihoods to likelihoods
# (or does it even matter?)
    f=exp(- A)
# calculate_jpdaf_beta_murty.m:14
    # Number of measurements Nm, number of tracks Nt
    Nm,Nt=size(A,nargout=2)
# calculate_jpdaf_beta_murty.m:17
    # Original distance matrix
    A_orig=copy(A)
# calculate_jpdaf_beta_murty.m:20
    # ########################################################### #
    
    #   Find N-best assignments using Murty's method
    
    # ########################################################### #
    
    # Initial problem
    P0=copy(A_orig)
# calculate_jpdaf_beta_murty.m:29
    # Initial optimal assignment
    S0,V0=munkres(P0,nargout=2)
# calculate_jpdaf_beta_murty.m:32
    SOLUTIONS=murtys_best(P0,S0,V0,1000)
# calculate_jpdaf_beta_murty.m:34
    if (length(SOLUTIONS) == 1):
        THETA[1]=S0
# calculate_jpdaf_beta_murty.m:39
    else:
        THETA=copy(SOLUTIONS)
# calculate_jpdaf_beta_murty.m:43
        THETA[end() + 1]=S0
# calculate_jpdaf_beta_murty.m:44
    
    num_events=length(THETA)
# calculate_jpdaf_beta_murty.m:48
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    for N in arange(1,num_events).reshape(-1):
        # Hadamard product of A_opt and f
        C_N=multiply(THETA[N],f)
# calculate_jpdaf_beta_murty.m:60
        theta_j,theta_t,v=find(C_N,nargout=3)
# calculate_jpdaf_beta_murty.m:63
        term1=prod(v)
# calculate_jpdaf_beta_murty.m:64
        term2=dot((PD) ** (length(theta_t)),(1 - PD) ** (Nt - length(theta_t)))
# calculate_jpdaf_beta_murty.m:67
        P_THETA[N]=dot(term1,term2)
# calculate_jpdaf_beta_murty.m:70
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# calculate_jpdaf_beta_murty.m:75
    # Normalize the probabilities
    for N in arange(1,num_events).reshape(-1):
        P_THETA[N]=P_THETA[N] / c
# calculate_jpdaf_beta_murty.m:80
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # For each target, calculate the probability of association
    
    for t in arange(1,Nt).reshape(-1):
        for j in arange(1,Nm).reshape(-1):
            beta[j,t]=0
# calculate_jpdaf_beta_murty.m:97
            for N in arange(1,num_events).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],THETA[N](j,t))
# calculate_jpdaf_beta_murty.m:101
    