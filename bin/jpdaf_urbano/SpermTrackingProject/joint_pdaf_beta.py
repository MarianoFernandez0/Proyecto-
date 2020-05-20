# Generated with SMOP  0.41
from libsmop import *
# joint_pdaf_beta.m

    
@function
def joint_beta_jpda(f=None,A=None,PD=None,*args,**kwargs):
    varargin = joint_beta_jpda.varargin
    nargin = joint_beta_jpda.nargin

    #   Sub-optimal Joint Probabilistic Data Association (JPDA)
#   Joint Association Probability Calculator
#   
#   Usage:  
#   
#       beta(j,t) = jpda_beta(f, A, PD)
    
    #
    
    # Number of measurements Nm, number of tracks Nt
    Nm,Nt=size(A,nargout=2)
# joint_pdaf_beta.m:13
    # Original distance matrix
    A_orig=copy(A)
# joint_pdaf_beta.m:16
    # ########################################################### #
    
    #   Determine Initial Optimal Assignment / Joint Event
    
    # ########################################################### #
    
    # Initial optimal assignment
    THETA[1]=munkres(A_orig)
# joint_pdaf_beta.m:25
    # How many Joint Events are there?
    num_events=length(find(THETA[1]))
# joint_pdaf_beta.m:28
    # ########################################################### #
    
    #   Calculate the N-best Joint Events
    
    # ########################################################### #
    
    # The set of events in the Initial Joint Event
    theta_j,theta_t=find(THETA[1],nargout=2)
# joint_pdaf_beta.m:38
    # Create N-1 new events
    for N in arange(1,num_events).reshape(-1):
        # Copy the Original Optimal Assignment
        A_TEMP=copy(A_orig)
# joint_pdaf_beta.m:44
        A_TEMP[theta_j(N),theta_t(N)]=Inf
# joint_pdaf_beta.m:47
        A_OPT=munkres(A_TEMP)
# joint_pdaf_beta.m:49
        THETA[end() + 1]=A_OPT
# joint_pdaf_beta.m:52
    
    # ########################################################### #
    
    #   Calculate the Probability of Each Joint Event
    
    # ########################################################### #
    
    for N in arange(1,(num_events + 1)).reshape(-1):
        # Hadamard product of A_opt and f
        C_N=multiply(THETA[N],f)
# joint_pdaf_beta.m:66
        theta_j,theta_t,v=find(C_N,nargout=3)
# joint_pdaf_beta.m:69
        term1=prod(v)
# joint_pdaf_beta.m:70
        term2=dot((PD) ** (length(theta_t)),(1 - PD) ** (Nt - length(theta_t)))
# joint_pdaf_beta.m:73
        P_THETA[N]=dot(term1,term2)
# joint_pdaf_beta.m:76
    
    # Calculate the normalization constant c
    c=sum(cell2mat(P_THETA))
# joint_pdaf_beta.m:81
    # Normalize the probabilities
    for N in arange(1,(num_events + 1)).reshape(-1):
        P_THETA[N]=P_THETA[N] / c
# joint_pdaf_beta.m:86
    
    # ########################################################### #
    
    #   Calculate the Joint Association Probability
    
    # ########################################################### #
    
    # For each target, calculate the probability of association
    
    for t in arange(1,Nt).reshape(-1):
        for j in arange(1,Nm).reshape(-1):
            beta[j,t]=0
# joint_pdaf_beta.m:103
            for N in arange(1,(num_events + 1)).reshape(-1):
                beta[j,t]=beta(j,t) + dot(P_THETA[N],THETA[N](j,t))
# joint_pdaf_beta.m:107
    