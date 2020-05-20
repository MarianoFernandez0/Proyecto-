# Generated with SMOP  0.41
from libsmop import *
# murtys_best.m

    
@function
def murtys_best(P0=None,S0=None,V0=None,N=None,*args,**kwargs):
    varargin = murtys_best.varargin
    nargin = murtys_best.nargin

    # 1. Find the best solution S0 to P0.
# [S0, V0] = munkres(P0);
    
    # 2. Initialize the list of problem / solution pairs with P0, S0, V0
    P[1]=P0
# murtys_best.m:6
    S[1]=S0
# murtys_best.m:6
    V[1]=V0
# murtys_best.m:6
    # 3. Clear the list of solutions to be returned
    SOLUTIONS=[]
# murtys_best.m:9
    VALUES=[]
# murtys_best.m:10
    # 4. For k = 1 to N (or until the list of P/S pairs is empty)
    for k in arange(1,N).reshape(-1):
        # 4.1 Find the solution with the best (minimum) value V
        best_idx=find(V == min(V),1)
# murtys_best.m:16
        if logical_not(isempty(best_idx)):
            P_BEST=P[best_idx]
# murtys_best.m:20
            S_BEST=S[best_idx]
# murtys_best.m:21
            V_BEST=V(best_idx)
# murtys_best.m:22
            P[best_idx]=[]
# murtys_best.m:25
            P[cellfun(lambda P=None: isempty(P),P)]=[]
# murtys_best.m:26
            S[best_idx]=[]
# murtys_best.m:27
            S[cellfun(lambda S=None: isempty(S),S)]=[]
# murtys_best.m:28
            V[best_idx]=[]
# murtys_best.m:29
            SOLUTIONS[end() + 1]=S_BEST
# murtys_best.m:32
            VALUES[end() + 1]=V_BEST
# murtys_best.m:33
            i,j=find(S_BEST,nargout=2)
# murtys_best.m:36
            for n in arange(1,length(i)).reshape(-1):
                # 4.4.1 Let P' = P
                P_PRIME=copy(P_BEST)
# murtys_best.m:40
                P_PRIME[i(n),j(n)]=Inf
# murtys_best.m:43
                S_PRIME,V_PRIME=munkres(P_PRIME,nargout=2)
# murtys_best.m:46
                if (sum(ravel(S_PRIME)) == length(i)):
                    # 4.4.4.1 Add <P',S'> to the set of P/S pairs
                    P[end() + 1]=P_PRIME
# murtys_best.m:52
                    S[end() + 1]=S_PRIME
# murtys_best.m:53
                    V[end() + 1]=V_PRIME
# murtys_best.m:54
                # 4.4.5 From P_BEST, clear the rows and columns from the n-th
            # assignment but leave the assignment intact
                a_ij=P_BEST(i(n),j(n))
# murtys_best.m:60
                P_BEST[i(n),arange()]=Inf
# murtys_best.m:61
                P_BEST[arange(),j(n)]=Inf
# murtys_best.m:62
                P_BEST[i(n),j(n)]=a_ij
# murtys_best.m:63
        else:
            break
    