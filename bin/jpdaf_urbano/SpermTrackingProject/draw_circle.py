# Generated with SMOP  0.41
from libsmop import *
# draw_circle.m

    
@function
def draw_circle(x0=None,y0=None,r=None,cc=None,*args,**kwargs):
    varargin = draw_circle.varargin
    nargin = draw_circle.nargin

    # x,y coordinates
# r radius
    
    x=x0 - r
# draw_circle.m:5
    y=y0 - r
# draw_circle.m:6
    w=dot(2,r)
# draw_circle.m:7
    h=dot(2,r)
# draw_circle.m:8
    g=rectangle('Position',concat([x,y,w,h]),'Curvature',concat([1,1]),'EdgeColor',cc)
# draw_circle.m:10