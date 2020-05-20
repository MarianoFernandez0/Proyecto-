# Generated with SMOP  0.41
from libsmop import *
# calculate_jpdaf_beta.m

    
@function
def calculate_jpdaf_beta(A=None,PD=None,*args,**kwargs):
    varargin = calculate_jpdaf_beta.varargin
    nargin = calculate_jpdaf_beta.nargin

    #   Sub-optimal Joint Probabilistic Data Association (JPDA)
#   Joint Association Probability Calculator
#   
#   Usage:  
#   
#       beta(j,t) = calculate_jpdaf_beta(f, A, PD)
    
    #
    
    # Convert from negative log likelihoods to likelihoods
# (or does it even matter?)
    f=exp(- A)
# calculate_jpdaf_beta.m:14
    # Number of measurements Nm, number of tracks Nt
    Nm,Nt=size(A,nargout=2)
# calculate_jpdaf_beta.m:17
    # Original distance matrix
    A_orig=copy(A)
# calculate_jpdaf_beta.m:20
    # ########################################################### #
    
    #   Determine Initial Optimal Assignment / Joint Event
    
    # ########################################################### #
    
    # Initial optimal assignment
    THETA[1]=munkres(A_orig)
# calculate_jpdaf_beta.m:29
    # How many Joint Events are there?
    num_events=length(find(THETA[1]))
# calculate_jpdaf_beta.m:32
    # ########################################################### #
    
    #   Calculate the N-best Joint Events
    
    # ########################################################### #
    
    # The set of events in the Initial Joint Event
    theta_j,theta_t=find(THETA[1],nargout=2)
# calculate_jpdaf_beta.m:42
    # Create N-1 new events
    for N in arange(1,num_events).reshape(-1):
        # Copy the Original Optimal Assignment
        A_TEMP=copy(A_orig)
# calculate_jpdaf_beta.m:48
        A_TEMP[theta_j(N),theta_t(N)]=Inf
# calculate_jpdaf_beta.m:51
        A_OPT=munkres(A_TEMP)
# calculate_jpdaf_beta.m:54
        THETA[end() + 1]=A_OPT
# calculate_jpdaf_beta.m:57
    
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    num_events=0
# calculate_jpdaf_beta.m:68
    for N in arange(1,(num_events + 1)).reshape(-1):
        # Hadamard product of A_opt and f
        C_N=multiply(THETA[N],f)
# calculate_jpdaf_beta.m:73
        theta_j,theta_t,v=find(C_N,nargout=3)
# calculate_jpdaf_beta.m:76
        term1=prod(v)
# calculate_jpdaf_beta.m:77
        term2=dot((PD) ** (length(theta_t)),(1 - PD) ** (Nt - length(theta_t)))
# calculate_jpdaf_beta.m:80
        P_THETA[N]=dot(term1,term2)
# calculate_jpdaf_beta.m:83
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# calculate_jpdaf_beta.m:88
    # Normalize the probabilities
    for N in arange(1,(num_events + 1)).reshape(-1):
        P_THETA[N]=P_THETA[N] / c
# calculate_jpdaf_beta.m:93
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # For each target, calculate the probability of association
    
    for t in arange(1,Nt).reshape(-1):
        for j in arange(1,Nm).reshape(-1):
            beta[j,t]=0
# calculate_jpdaf_beta.m:110
            for N in arange(1,(num_events + 1)).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],THETA[N](j,t))
# calculate_jpdaf_beta.m:114
    