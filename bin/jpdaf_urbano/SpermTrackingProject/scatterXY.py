# Generated with SMOP  0.41
from libsmop import *
# scatterXY.m

    
@function
def scatterXY(X=None,Y=None,numBins=None,onOff=None,*args,**kwargs):
    varargin = scatterXY.varargin
    nargin = scatterXY.nargin

    n,c=hist3(concat([X.T,Y.T]),concat([numBins,numBins]),nargout=2)
# scatterXY.m:3
    d=n / max(max(n))
# scatterXY.m:4
    e=interp2(concat([c[1]]),concat([c[2]]),d.T,X,Y)
# scatterXY.m:5
    __,b=sort(e,'descend',nargout=2)
# scatterXY.m:6
    scatter(X(b),Y(b),20,e(b),'filled')
    if (onOff):
        cHandle=copy(colorbar)
# scatterXY.m:9
        ylabel(cHandle,'Relative Density of Data Points','FontSize',12,'FontName','Arial','FontWeight','Bold')
        caxis(concat([0,1]))
        set(cHandle,'YTick',[])
        colormap('parula')
    