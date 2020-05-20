# Generated with SMOP  0.41
from libsmop import *
# calculate_beta_jpda.m

    
@function
def calculate_beta_jpda(A=None,*args,**kwargs):
    varargin = calculate_beta_jpda.varargin
    nargin = calculate_beta_jpda.nargin

    #   Sub-optimal Joint Probabilistic Data Association (JPDA)
#   Joint Association Probability Calculator using murty's k-best
#   
#   Usage:  
#   
#       beta(j,t) = calculate_jpdaf_beta_murty(f, A, PD)
    
    #
    
    PD=0.95
# calculate_beta_jpda.m:11
    # Convert from negative log likelihoods to likelihoods
# (or does it even matter?)
    f=exp(- A)
# calculate_beta_jpda.m:15
    # Number of measurements Nm, number of tracks Nt
    Nm,Nt=size(A,nargout=2)
# calculate_beta_jpda.m:18
    # Original distance matrix
    A_orig=copy(A)
# calculate_beta_jpda.m:21
    # ########################################################### #
    
    #   Find N-best assignments using Murty's method
    
    # ########################################################### #
    
    # Initial problem
    P0=copy(A_orig)
# calculate_beta_jpda.m:30
    # Initial optimal assignment
    S0,V0=munkres(P0,nargout=2)
# calculate_beta_jpda.m:33
    SOLUTIONS=murtys_best(P0,S0,V0,10)
# calculate_beta_jpda.m:35
    if (length(SOLUTIONS) == 1):
        THETA[1]=S0
# calculate_beta_jpda.m:39
    else:
        THETA=copy(SOLUTIONS)
# calculate_beta_jpda.m:43
        THETA[end() + 1]=S0
# calculate_beta_jpda.m:44
    
    num_events=length(THETA)
# calculate_beta_jpda.m:48
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    for N in arange(1,num_events).reshape(-1):
        # Hadamard product of A_opt and f
        C_N=multiply(THETA[N],f)
# calculate_beta_jpda.m:60
        # P{ THETA | z^k}
        theta_j,theta_t,v=find(C_N,nargout=3)
# calculate_beta_jpda.m:64
        term1=sum(- log(v))
# calculate_beta_jpda.m:65
        term2=dot(length(theta_t),log(PD)) + dot((Nt - length(theta_t)),log(1 - PD))
# calculate_beta_jpda.m:68
        # Probability of joint event
        P_THETA[N]=term1 + term2
# calculate_beta_jpda.m:72
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# calculate_beta_jpda.m:77
    # Normalize the probabilities
    for N in arange(1,num_events).reshape(-1):
        P_THETA[N]=P_THETA[N] - c
# calculate_beta_jpda.m:82
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # For each target, calculate the probability of association
    
    for t in arange(1,Nt).reshape(-1):
        for j in arange(1,Nm).reshape(-1):
            beta[j,t]=0
# calculate_beta_jpda.m:99
            for N in arange(1,num_events).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],THETA[N](j,t))
# calculate_beta_jpda.m:103
    