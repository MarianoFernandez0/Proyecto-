# Generated with SMOP  0.41
from libsmop import *
# localmax.m

    
@function
def localmax(x=None,w=None,*args,**kwargs):
    varargin = localmax.varargin
    nargin = localmax.nargin

    # LOCALMAX  Find indices and amplitudes of local maxima 
# [m,i] = localmax(x, w) returns the indices and maxima defined 
# over a local window of size 2w+1 given by w points on either 
# side of the point being considered as a local maximum..
    
    # by P.E.McSharry
# These routines are made available under the GNU general public license. 
# If you have not received a copy of this license, please download from 
# http://www.gnu.org/
    
    # Please distribute (and modify) freely, commenting where you have 
# added modifications. The author would appreciate correspondence 
# regarding corrections, modifications, improvements etc.
    
    # G. Clifford : gari@ieee.org
    
    N=length(x)
# localmax.m:19
    k=dot(2,w) + 1
# localmax.m:21
    y=zeros(k,1)
# localmax.m:22
    l=0
# localmax.m:24
    for j in arange(w + 1,N - w).reshape(-1):
        y=x(arange(j - w,j + w))
# localmax.m:26
        ymax,imax=max(y,nargout=2)
# localmax.m:27
        if imax == w + 1:
            l=l + 1
# localmax.m:29
            m[l]=ymax
# localmax.m:30
            i[l]=j
# localmax.m:31
    