# Generated with SMOP  0.41
from libsmop import *
# plot_data_pdfs.m

    
@function
def plot_data_pdfs(P=None,Q=None,*args,**kwargs):
    varargin = plot_data_pdfs.varargin
    nargin = plot_data_pdfs.nargin

    fP,xP=hist(P,100,nargout=2)
# plot_data_pdfs.m:3
    plot(xP,fP / trapz(xP,fP),'b','LineWidth',1)
    fQ,xQ=hist(Q,100,nargout=2)
# plot_data_pdfs.m:6
    plot(xQ,fQ / trapz(xQ,fQ),'r--','LineWidth',1)