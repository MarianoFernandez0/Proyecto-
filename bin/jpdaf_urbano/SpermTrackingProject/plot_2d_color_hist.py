# Generated with SMOP  0.41
from libsmop import *
# plot_2d_color_hist.m

    
@function
def plot_lin_alh_hist(SAMPLE_ALH=None,SAMPLE_LIN=None,*args,**kwargs):
    varargin = plot_lin_alh_hist.varargin
    nargin = plot_lin_alh_hist.nargin

    # Plots the 2D histogram of ALH and LIN
    
    h=copy(figure)
# plot_2d_color_hist.m:5
    hold('on')
    grid('on')
    axis(concat([0,20,0,1]))
    # ####################################################################### #
    
    #   Label the histogram
    
    # ####################################################################### #
    
    title(concat(['Kinematic Properties of the Sample']),'FontSize',12,'FontName','Arial','FontWeight','Bold')
    xlabel('Amplitude Lateral Head Displacement ALH (\mum)','FontSize',12,'FontName','Arial','FontWeight','Bold')
    ylabel('Linearity VSL /VCL (unitless)','FontSize',12,'FontName','Arial','FontWeight','Bold')
    set(gca,'FontSize',12,'FontName','Arial','FontWeight','Bold')
    # ####################################################################### #
    
    #   Draw the colorbar
    
    # ####################################################################### #
    
    # Draw the colorbar
    colorbar_handle=copy(colorbar)
# plot_2d_color_hist.m:37
    # Label the colorbar
    ylabel(colorbar_handle,'Relative Density of Data Points','FontSize',12,'FontName','Arial','FontWeight','Bold')
    # Set the colorbar limits
    caxis(concat([0,1]))
    # Generate histogram data the Scatter Plot
    NN,CC=hist3(concat([SAMPLE_ALH.T,SAMPLE_LIN.T]),concat([25,25]),nargout=2)
# plot_2d_color_hist.m:47
    # Normalize the colors
    DD=NN / (max(max(NN)))
# plot_2d_color_hist.m:50
    # Interpolate the data matrix
    EE=interp2(CC[1],CC[2],DD.T,SAMPLE_ALH,SAMPLE_LIN)
# plot_2d_color_hist.m:53
    # Sort the data so red is on top
    gg,II=sort(EE,'descend',nargout=2)
# plot_2d_color_hist.m:56
    # Draw the scatter plot
    plotHandles[end() + 1]=scatter(SAMPLE_ALH(II),SAMPLE_LIN(II),20,EE(II),'filled')
# plot_2d_color_hist.m:59