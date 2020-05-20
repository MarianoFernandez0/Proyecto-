# Generated with SMOP  0.41
from libsmop import *
# trk_ospa_dist.m

    
@function
def trk_ospa_dist(X=None,Xlab=None,Y=None,Ylab=None,OSPA=None,*args,**kwargs):
    varargin = trk_ospa_dist.varargin
    nargin = trk_ospa_dist.nargin

    #B. Vo.  26/08/2007
#Compute Schumacher distance between two finite sets X and Y
#Inputs: X,Y-   matrices of column vectors
#        c  -   cut-off parameter
#        p  -   p-parameter for the metric
#Output: scalar distance between X and Y
#Note: the Euclidean 2-norm is used as the "base" distance on the region
    
    # Modified by Urbano for 2D pos vectors
    
    
    p=OSPA.p
# trk_ospa_dist.m:15
    c=OSPA.c
# trk_ospa_dist.m:16
    l=OSPA.l
# trk_ospa_dist.m:17
    if (nargout != 1) and (nargout != 3):
        error('Incorrect number of outputs')
    
    #Calculate sizes of the input point patterns
    n=size(X,2)
# trk_ospa_dist.m:24
    m=size(Y,2)
# trk_ospa_dist.m:25
    if (n == 0) and (m == 0):
        dist=0
# trk_ospa_dist.m:28
        if nargout == 3:
            varargout[1]=cellarray([0])
# trk_ospa_dist.m:30
            varargout[2]=cellarray([0])
# trk_ospa_dist.m:31
        return dist,varargout
    
    if (n == 0) or (m == 0):
        dist=(dot(dot(1 / max(m,n),c ** p),abs(m - n))) ** (1 / p)
# trk_ospa_dist.m:37
        if nargout == 3:
            varargout[1]=cellarray([0])
# trk_ospa_dist.m:39
            varargout[2]=cellarray([(dot(dot(1 / max(m,n),c ** p),abs(m - n))) ** (1 / p)])
# trk_ospa_dist.m:40
        return dist,varargout
    
    #Calculate cost/weight matrix for pairings - fast method with vectorization
# XX= repmat(X,[1 m]);
# YY= reshape(repmat(Y,[n 1]),[size(Y,1) n*m]);
# D = reshape(sqrt(sum((XX-YY).^2)),[n m]);
# D = min(c,D).^p;
    
    # Calculate cost/weight matrix for pairings - slow method with for loop
    D=zeros(n,m)
# trk_ospa_dist.m:53
    for j in arange(1,m).reshape(-1):
        for i in arange(1,n).reshape(-1):
            bdist=sum(abs(Y(concat([1,2]),j) - X(concat([1,2]),i)) ** p)
# trk_ospa_dist.m:56
            ldist=dot(OSPA.l ** p,not_(Xlab(i) == Ylab(j)))
# trk_ospa_dist.m:57
            D[i,j]=(bdist + ldist) ** (1 / p)
# trk_ospa_dist.m:58
    
    D=min(c,D) ** p
# trk_ospa_dist.m:61
    #Compute optimal assignment and cost using the Hungarian algorithm
    assignment,cost=munkres(D,nargout=2)
# trk_ospa_dist.m:64
    # # assignment based on labels
# iass = zeros(n,m);
# for i=1:n
#     ix = find( Xlab(i) == Ylab);
#     if not(isempty(ix))
#         iass(i,ix) = 1;
#     end
# end
# label_error = sum(sum(abs(assignment - iass)));
    
    #Calculate final distance
    dist=(dot(1 / max(m,n),(dot(c ** p,abs(m - n)) + cost))) ** (1 / p)
# trk_ospa_dist.m:77
    #Output components if called for in varargout
    if nargout == 3:
        varargout[1]=cellarray([(dot(1 / max(m,n),cost)) ** (1 / p)])
# trk_ospa_dist.m:81
        varargout[2]=cellarray([(dot(dot(1 / max(m,n),c ** p),abs(m - n))) ** (1 / p)])
# trk_ospa_dist.m:82
    
    