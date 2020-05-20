# Generated with SMOP  0.41
from libsmop import *
# plot_gaussian_ellipsoid.m

    
@function
def plot_gaussian_ellipsoid(m=None,C=None,sdwidth=None,npts=None,axh=None,*args,**kwargs):
    varargin = plot_gaussian_ellipsoid.varargin
    nargin = plot_gaussian_ellipsoid.nargin

    # PLOT_GAUSSIAN_ELLIPSOIDS plots 2-d and 3-d Gaussian distributions
    
    # H = PLOT_GAUSSIAN_ELLIPSOIDS(M, C) plots the distribution specified by 
#  mean M and covariance C. The distribution is plotted as an ellipse (in 
#  2-d) or an ellipsoid (in 3-d).  By default, the distributions are 
#  plotted in the current axes. H is the graphics handle to the plotted 
#  ellipse or ellipsoid.
    
    # PLOT_GAUSSIAN_ELLIPSOIDS(M, C, SD) uses SD as the standard deviation 
#  along the major and minor axes (larger SD => larger ellipse). By 
#  default, SD = 1. Note: 
#  * For 2-d distributions, SD=1.0 and SD=2.0 cover ~ 39# and 86# 
#     of the total probability mass, respectively. 
#  * For 3-d distributions, SD=1.0 and SD=2.0 cover ~ 19# and 73#
#     of the total probability mass, respectively.
#  
# PLOT_GAUSSIAN_ELLIPSOIDS(M, C, SD, NPTS) plots the ellipse or 
#  ellipsoid with a resolution of NPTS (ellipsoids are generated 
#  on an NPTS x NPTS mesh; see SPHERE for more details). By
#  default, NPTS = 50 for ellipses, and 20 for ellipsoids.
    
    # PLOT_GAUSSIAN_ELLIPSOIDS(M, C, SD, NPTS, AX) adds the plot to the
#  axes specified by the axis handle AX.
    
    # Examples: 
# -------------------------------------------
#  # Plot three 2-d Gaussians
#  figure; 
#  h1 = plot_gaussian_ellipsoid([1 1], [1 0.5; 0.5 1]);
#  h2 = plot_gaussian_ellipsoid([2 1.5], [1 -0.7; -0.7 1]);
#  h3 = plot_gaussian_ellipsoid([0 0], [1 0; 0 1]);
#  set(h2,'color','r'); 
#  set(h3,'color','g');
# 
#  # "Contour map" of a 2-d Gaussian
#  figure;
#  for sd = [0.3:0.4:4],
#    h = plot_gaussian_ellipsoid([0 0], [1 0.8; 0.8 1], sd);
#  end
    
    #  # Plot three 3-d Gaussians
#  figure;
#  h1 = plot_gaussian_ellipsoid([1 1  0], [1 0.5 0.2; 0.5 1 0.4; 0.2 0.4 1]);
#  h2 = plot_gaussian_ellipsoid([1.5 1 .5], [1 -0.7 0.6; -0.7 1 0; 0.6 0 1]);
#  h3 = plot_gaussian_ellipsoid([1 2 2], [0.5 0 0; 0 0.5 0; 0 0 0.5]);
#  set(h2,'facealpha',0.6);
#  view(129,36); set(gca,'proj','perspective'); grid on; 
#  grid on; axis equal; axis tight;
# -------------------------------------------
# 
#  Gautam Vallabha, Sep-23-2007, Gautam.Vallabha@mathworks.com
    
    #  Revision 1.0, Sep-23-2007
#    - File created
#  Revision 1.1, 26-Sep-2007
#    - NARGOUT==0 check added.
#    - Help added on NPTS for ellipsoids
    
    if logical_not(exist('sdwidth','var')):
        sdwidth=1
# plot_gaussian_ellipsoid.m:60
    
    if logical_not(exist('npts','var')):
        npts=[]
# plot_gaussian_ellipsoid.m:61
    
    if logical_not(exist('axh','var')):
        axh=copy(gca)
# plot_gaussian_ellipsoid.m:62
    
    if numel(m) != length(m):
        error('M must be a vector')
    
    if logical_not((all(numel(m) == size(C)))):
        error('Dimensionality of M and C must match')
    
    if logical_not((isscalar(axh) and ishandle(axh) and strcmp(get(axh,'type'),'axes'))):
        error('Invalid axes handle')
    
    set(axh,'nextplot','add')
    if 2 == numel(m):
        h=show2d(ravel(m),C,sdwidth,npts,axh)
# plot_gaussian_ellipsoid.m:77
    else:
        if 3 == numel(m):
            h=show3d(ravel(m),C,sdwidth,npts,axh)
# plot_gaussian_ellipsoid.m:78
        else:
            error('Unsupported dimensionality')
    
    if nargout == 0:
        clear('h')
    
    #-----------------------------
    
@function
def show2d(means=None,C=None,sdwidth=None,npts=None,axh=None,*args,**kwargs):
    varargin = show2d.varargin
    nargin = show2d.nargin

    if isempty(npts):
        npts=50
# plot_gaussian_ellipsoid.m:89
    
    # plot the gaussian fits
    tt=linspace(0,dot(2,pi),npts).T
# plot_gaussian_ellipsoid.m:91
    x=cos(tt)
# plot_gaussian_ellipsoid.m:92
    y=sin(tt)
# plot_gaussian_ellipsoid.m:92
    ap=concat([ravel(x),ravel(y)]).T
# plot_gaussian_ellipsoid.m:93
    v,d=eig(C,nargout=2)
# plot_gaussian_ellipsoid.m:94
    d=dot(sdwidth,sqrt(d))
# plot_gaussian_ellipsoid.m:95
    
    bp=(dot(dot(v,d),ap)) + repmat(means,1,size(ap,2))
# plot_gaussian_ellipsoid.m:96
    h=plot(bp(1,arange()),bp(2,arange()),'-','parent',axh)
# plot_gaussian_ellipsoid.m:97
    #-----------------------------
    
@function
def show3d(means=None,C=None,sdwidth=None,npts=None,axh=None,*args,**kwargs):
    varargin = show3d.varargin
    nargin = show3d.nargin

    if isempty(npts):
        npts=20
# plot_gaussian_ellipsoid.m:101
    
    x,y,z=sphere(npts,nargout=3)
# plot_gaussian_ellipsoid.m:102
    ap=concat([ravel(x),ravel(y),ravel(z)]).T
# plot_gaussian_ellipsoid.m:103
    v,d=eig(C,nargout=2)
# plot_gaussian_ellipsoid.m:104
    if any(ravel(d) < 0):
        fprintf('warning: negative eigenvalues\n')
        d=max(d,0)
# plot_gaussian_ellipsoid.m:107
    
    d=dot(sdwidth,sqrt(d))
# plot_gaussian_ellipsoid.m:109
    
    bp=(dot(dot(v,d),ap)) + repmat(means,1,size(ap,2))
# plot_gaussian_ellipsoid.m:110
    xp=reshape(bp(1,arange()),size(x))
# plot_gaussian_ellipsoid.m:111
    yp=reshape(bp(2,arange()),size(y))
# plot_gaussian_ellipsoid.m:112
    zp=reshape(bp(3,arange()),size(z))
# plot_gaussian_ellipsoid.m:113
    h=surf(axh,xp,yp,zp)
# plot_gaussian_ellipsoid.m:114