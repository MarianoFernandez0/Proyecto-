# Generated with SMOP  0.41
from libsmop import *
# Hungarian.m

    
@function
def Hungarian(Perf=None,*args,**kwargs):
    varargin = Hungarian.varargin
    nargin = Hungarian.nargin

    # 
# [MATCHING,COST] = Hungarian_New(WEIGHTS)
    
    # A function for finding a minimum edge weight matching given a MxN Edge
# weight matrix WEIGHTS using the Hungarian Algorithm.
    
    # An edge weight of Inf indicates that the pair of vertices given by its
# position have no adjacent edge.
    
    # MATCHING return a MxN matrix with ones in the place of the matchings and
# zeros elsewhere.
# 
# COST returns the cost of the minimum matching
    
    # Written by: Alex Melin 30 June 2006
    
    # Initialize Variables
    Matching=zeros(size(Perf))
# Hungarian.m:20
    # Condense the Performance Matrix by removing any unconnected vertices to
# increase the speed of the algorithm
    
    # Find the number in each column that are connected
    num_y=sum(logical_not(isinf(Perf)),1)
# Hungarian.m:26
    
    num_x=sum(logical_not(isinf(Perf)),2)
# Hungarian.m:28
    
    x_con=find(num_x != 0)
# Hungarian.m:31
    y_con=find(num_y != 0)
# Hungarian.m:32
    
    P_size=max(length(x_con),length(y_con))
# Hungarian.m:35
    P_cond=zeros(P_size)
# Hungarian.m:36
    P_cond[arange(1,length(x_con)),arange(1,length(y_con))]=Perf(x_con,y_con)
# Hungarian.m:37
    if isempty(P_cond):
        Cost=0
# Hungarian.m:39
        return Matching,Cost
    
    # Ensure that a perfect matching exists
      # Calculate a form of the Edge Matrix
    Edge=copy(P_cond)
# Hungarian.m:45
    Edge[P_cond != Inf]=0
# Hungarian.m:46
    
    cnum=min_line_cover(Edge)
# Hungarian.m:48
    
    # exists
    Pmax=max(max(P_cond(P_cond != Inf)))
# Hungarian.m:52
    P_size=length(P_cond) + cnum
# Hungarian.m:53
    P_cond=dot(ones(P_size),Pmax)
# Hungarian.m:54
    P_cond[arange(1,length(x_con)),arange(1,length(y_con))]=Perf(x_con,y_con)
# Hungarian.m:55
    #*************************************************
# MAIN PROGRAM: CONTROLS WHICH STEP IS EXECUTED
#*************************************************
    exit_flag=1
# Hungarian.m:60
    stepnum=1
# Hungarian.m:61
    while exit_flag:

        if 1 == stepnum:
            P_cond,stepnum=step1(P_cond,nargout=2)
# Hungarian.m:65
        else:
            if 2 == stepnum:
                r_cov,c_cov,M,stepnum=step2(P_cond,nargout=4)
# Hungarian.m:67
            else:
                if 3 == stepnum:
                    c_cov,stepnum=step3(M,P_size,nargout=2)
# Hungarian.m:69
                else:
                    if 4 == stepnum:
                        M,r_cov,c_cov,Z_r,Z_c,stepnum=step4(P_cond,r_cov,c_cov,M,nargout=6)
# Hungarian.m:71
                    else:
                        if 5 == stepnum:
                            M,r_cov,c_cov,stepnum=step5(M,Z_r,Z_c,r_cov,c_cov,nargout=4)
# Hungarian.m:73
                        else:
                            if 6 == stepnum:
                                P_cond,stepnum=step6(P_cond,r_cov,c_cov,nargout=2)
# Hungarian.m:75
                            else:
                                if 7 == stepnum:
                                    exit_flag=0
# Hungarian.m:77

    
    # Remove all the virtual satellites and targets and uncondense the
# Matching to the size of the original performance matrix.
    Matching[x_con,y_con]=M(arange(1,length(x_con)),arange(1,length(y_con)))
# Hungarian.m:83
    Cost=sum(sum(Perf(Matching == 1)))
# Hungarian.m:84
    #########################################################
#   STEP 1: Find the smallest number of zeros in each row
#           and subtract that minimum from its row
#########################################################
    
    
@function
def step1(P_cond=None,*args,**kwargs):
    varargin = step1.varargin
    nargin = step1.nargin

    P_size=length(P_cond)
# Hungarian.m:93
    
    for ii in arange(1,P_size).reshape(-1):
        rmin=min(P_cond(ii,arange()))
# Hungarian.m:97
        P_cond[ii,arange()]=P_cond(ii,arange()) - rmin
# Hungarian.m:98
    
    stepnum=2
# Hungarian.m:101
    #**************************************************************************  
#   STEP 2: Find a zero in P_cond. If there are no starred zeros in its
#           column or row start the zero. Repeat for each zero
#**************************************************************************
    
    
@function
def step2(P_cond=None,*args,**kwargs):
    varargin = step2.varargin
    nargin = step2.nargin

    # Define variables
    P_size=length(P_cond)
# Hungarian.m:111
    r_cov=zeros(P_size,1)
# Hungarian.m:112
    
    c_cov=zeros(P_size,1)
# Hungarian.m:113
    
    M=zeros(P_size)
# Hungarian.m:114
    
    
    for ii in arange(1,P_size).reshape(-1):
        for jj in arange(1,P_size).reshape(-1):
            if P_cond(ii,jj) == 0 and r_cov(ii) == 0 and c_cov(jj) == 0:
                M[ii,jj]=1
# Hungarian.m:119
                r_cov[ii]=1
# Hungarian.m:120
                c_cov[jj]=1
# Hungarian.m:121
    
    
    # Re-initialize the cover vectors
    r_cov=zeros(P_size,1)
# Hungarian.m:127
    
    c_cov=zeros(P_size,1)
# Hungarian.m:128
    
    stepnum=3
# Hungarian.m:129
    #**************************************************************************
#   STEP 3: Cover each column with a starred zero. If all the columns are
#           covered then the matching is maximum
#**************************************************************************
    
    
@function
def step3(M=None,P_size=None,*args,**kwargs):
    varargin = step3.varargin
    nargin = step3.nargin

    c_cov=sum(M,1)
# Hungarian.m:138
    if sum(c_cov) == P_size:
        stepnum=7
# Hungarian.m:140
    else:
        stepnum=4
# Hungarian.m:142
    
    
    #**************************************************************************
#   STEP 4: Find a noncovered zero and prime it.  If there is no starred
#           zero in the row containing this primed zero, Go to Step 5.  
#           Otherwise, cover this row and uncover the column containing 
#           the starred zero. Continue in this manner until there are no 
#           uncovered zeros left. Save the smallest uncovered value and 
#           Go to Step 6.
#**************************************************************************
    
@function
def step4(P_cond=None,r_cov=None,c_cov=None,M=None,*args,**kwargs):
    varargin = step4.varargin
    nargin = step4.nargin

    P_size=length(P_cond)
# Hungarian.m:155
    zflag=1
# Hungarian.m:157
    while zflag:

        # Find the first uncovered zero
        row=0
# Hungarian.m:160
        col=0
# Hungarian.m:160
        exit_flag=1
# Hungarian.m:160
        ii=1
# Hungarian.m:161
        jj=1
# Hungarian.m:161
        while exit_flag:

            if P_cond(ii,jj) == 0 and r_cov(ii) == 0 and c_cov(jj) == 0:
                row=copy(ii)
# Hungarian.m:164
                col=copy(jj)
# Hungarian.m:165
                exit_flag=0
# Hungarian.m:166
            jj=jj + 1
# Hungarian.m:168
            if jj > P_size:
                jj=1
# Hungarian.m:169
                ii=ii + 1
# Hungarian.m:169
            if ii > P_size:
                exit_flag=0
# Hungarian.m:170

        # If there are no uncovered zeros go to step 6
        if row == 0:
            stepnum=6
# Hungarian.m:175
            zflag=0
# Hungarian.m:176
            Z_r=0
# Hungarian.m:177
            Z_c=0
# Hungarian.m:178
        else:
            # Prime the uncovered zero
            M[row,col]=2
# Hungarian.m:181
            # Cover the row and uncover the column containing the zero
            if sum(find(M(row,arange()) == 1)) != 0:
                r_cov[row]=1
# Hungarian.m:185
                zcol=find(M(row,arange()) == 1)
# Hungarian.m:186
                c_cov[zcol]=0
# Hungarian.m:187
            else:
                stepnum=5
# Hungarian.m:189
                zflag=0
# Hungarian.m:190
                Z_r=copy(row)
# Hungarian.m:191
                Z_c=copy(col)
# Hungarian.m:192

    
    
    #**************************************************************************
# STEP 5: Construct a series of alternating primed and starred zeros as
#         follows.  Let Z0 represent the uncovered primed zero found in Step 4.
#         Let Z1 denote the starred zero in the column of Z0 (if any). 
#         Let Z2 denote the primed zero in the row of Z1 (there will always
#         be one).  Continue until the series terminates at a primed zero
#         that has no starred zero in its column.  Unstar each starred 
#         zero of the series, star each primed zero of the series, erase 
#         all primes and uncover every line in the matrix.  Return to Step 3.
#**************************************************************************
    
    
@function
def step5(M=None,Z_r=None,Z_c=None,r_cov=None,c_cov=None,*args,**kwargs):
    varargin = step5.varargin
    nargin = step5.nargin

    zflag=1
# Hungarian.m:210
    ii=1
# Hungarian.m:211
    while zflag:

        # Find the index number of the starred zero in the column
        rindex=find(M(arange(),Z_c(ii)) == 1)
# Hungarian.m:214
        if rindex > 0:
            # Save the starred zero
            ii=ii + 1
# Hungarian.m:217
            Z_r[ii,1]=rindex
# Hungarian.m:219
            # primed zero
            Z_c[ii,1]=Z_c(ii - 1)
# Hungarian.m:222
        else:
            zflag=0
# Hungarian.m:224
        # Continue if there is a starred zero in the column of the primed zero
        if zflag == 1:
            cindex=find(M(Z_r(ii),arange()) == 2)
# Hungarian.m:230
            ii=ii + 1
# Hungarian.m:231
            Z_r[ii,1]=Z_r(ii - 1)
# Hungarian.m:232
            Z_c[ii,1]=cindex
# Hungarian.m:233

    
    
    # UNSTAR all the starred zeros in the path and STAR all primed zeros
    for ii in arange(1,length(Z_r)).reshape(-1):
        if M(Z_r(ii),Z_c(ii)) == 1:
            M[Z_r(ii),Z_c(ii)]=0
# Hungarian.m:240
        else:
            M[Z_r(ii),Z_c(ii)]=1
# Hungarian.m:242
    
    
    # Clear the covers
    r_cov=multiply(r_cov,0)
# Hungarian.m:247
    c_cov=multiply(c_cov,0)
# Hungarian.m:248
    
    M[M == 2]=0
# Hungarian.m:251
    stepnum=3
# Hungarian.m:253
    # *************************************************************************
# STEP 6: Add the minimum uncovered value to every element of each covered
#         row, and subtract it from every element of each uncovered column.  
#         Return to Step 4 without altering any stars, primes, or covered lines.
#**************************************************************************
    
    
@function
def step6(P_cond=None,r_cov=None,c_cov=None,*args,**kwargs):
    varargin = step6.varargin
    nargin = step6.nargin

    a=find(r_cov == 0)
# Hungarian.m:262
    b=find(c_cov == 0)
# Hungarian.m:263
    minval=min(min(P_cond(a,b)))
# Hungarian.m:264
    P_cond[find(r_cov == 1),arange()]=P_cond(find(r_cov == 1),arange()) + minval
# Hungarian.m:266
    P_cond[arange(),find(c_cov == 0)]=P_cond(arange(),find(c_cov == 0)) - minval
# Hungarian.m:267
    stepnum=4
# Hungarian.m:269
    
@function
def min_line_cover(Edge=None,*args,**kwargs):
    varargin = min_line_cover.varargin
    nargin = min_line_cover.nargin

    # Step 2
    r_cov,c_cov,M,stepnum=step2(Edge,nargout=4)
# Hungarian.m:274
    
    c_cov,stepnum=step3(M,length(Edge),nargout=2)
# Hungarian.m:276
    
    M,r_cov,c_cov,Z_r,Z_c,stepnum=step4(Edge,r_cov,c_cov,M,nargout=6)
# Hungarian.m:278
    
    cnum=length(Edge) - sum(r_cov) - sum(c_cov)
# Hungarian.m:280