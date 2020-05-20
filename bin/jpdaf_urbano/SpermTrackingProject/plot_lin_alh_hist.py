# Generated with SMOP  0.41
from libsmop import *
# plot_lin_alh_hist.m

    
@function
def plot_lin_alh_hist(SAMPLE_ALH=None,SAMPLE_LIN=None,*args,**kwargs):
    varargin = plot_lin_alh_hist.varargin
    nargin = plot_lin_alh_hist.nargin

    # Plots the 2D histogram of ALH and LIN
# SAMPLE_ALH = [SAMPLE_ALH -0.01 20.01];
# SAMPLE_LIN = [SAMPLE_LIN -0.01 1.01];
    
    # figure; hold on; grid on;
    axis(concat([0,20,0,1]))
    set(gcf,'PaperPositionMode','Auto')
    # ####################################################################### #
    
    #   Label the histogram
    
    # ####################################################################### #
    
    # title(['Kinematic Properties of the Sample'], 'FontSize', 12, ...
#     'FontName', 'Arial', 'FontWeight', 'Bold');
    xlabel('ALH (\mum)','FontSize',18,'FontName','Arial','FontWeight','Bold')
    ylabel('LIN = VCL/VSL','FontSize',18,'FontName','Arial','FontWeight','Bold')
    set(gca,'FontSize',18,'FontName','Arial','FontWeight','Bold','Box','On')
    # ####################################################################### #
    
    #   Draw colorbar
    
    # ####################################################################### #
    
    colorbar_handle=copy(colorbar)
# plot_lin_alh_hist.m:33
    ylabel(colorbar_handle,'Relative Density of Data Points','FontSize',18,'FontName','Arial','FontWeight','Bold')
    caxis(concat([0,1]))
    # ####################################################################### #
    
    #   Normalize the data
    
    # ####################################################################### #
    
    NN,CC=hist3(concat([SAMPLE_ALH.T,SAMPLE_LIN.T]),concat([25,25]),nargout=2)
# plot_lin_alh_hist.m:45
    DD=NN / (max(max(NN)))
# plot_lin_alh_hist.m:46
    # Interpolate the data matrix
    EE=interp2(CC[1],CC[2],DD.T,SAMPLE_ALH,SAMPLE_LIN)
# plot_lin_alh_hist.m:49
    # Sort the data so red is on top
    gg,II=sort(EE,'descend',nargout=2)
# plot_lin_alh_hist.m:52
    # Draw the scatter plot
    scatter(SAMPLE_ALH(II),SAMPLE_LIN(II),20,EE(II),'filled')