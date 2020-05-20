# Generated with SMOP  0.41
from libsmop import *
# plot_vcl_vsl_hist.m

    
@function
def plot_vcl_vsl_hist(SAMPLE_VSL=None,SAMPLE_VCL=None,*args,**kwargs):
    varargin = plot_vcl_vsl_hist.varargin
    nargin = plot_vcl_vsl_hist.nargin

    # Plots the 2D histogram of VSL and VSL
# SAMPLE_VCL = [SAMPLE_VCL -0.01 250.01];
# SAMPLE_VSL = [SAMPLE_VSL -0.01 150.01];
    
    # h = figure; hold on; grid on;
    axis(concat([0,150,0,250]))
    # set(gcf, 'PaperPositionMode', 'Auto')
    
    # ####################################################################### #
    
    #   Label the histogram
    
    # ####################################################################### #
    
    # title(['Kinematic Properties of the Sample'], 'FontSize', 12, ...
#     'FontName', 'Arial', 'FontWeight', 'Bold');
    xlabel('VSL (\mum/s)','FontSize',18,'FontName','Arial','FontWeight','Bold')
    ylabel('VCL (\mum/s)','FontSize',18,'FontName','Arial','FontWeight','Bold')
    set(gca,'FontSize',14,'FontName','Arial','FontWeight','Bold','Box','On')
    # ####################################################################### #
    
    #   Draw colorbar
    
    # ####################################################################### #
    colorbar_handle=copy(colorbar)
# plot_vcl_vsl_hist.m:31
    ylabel(colorbar_handle,'Relative Density of Data Points','FontSize',18,'FontName','Arial','FontWeight','Bold')
    caxis(concat([0,1]))
    # ####################################################################### #
    
    #  Normalize the data
    
    # ####################################################################### #
    
    NN,CC=hist3(concat([SAMPLE_VSL.T,SAMPLE_VCL.T]),concat([25,25]),nargout=2)
# plot_vcl_vsl_hist.m:43
    DD=NN / (max(max(NN)))
# plot_vcl_vsl_hist.m:44
    # Interpolate the data matrix
    EE=interp2(CC[1],CC[2],DD.T,concat([SAMPLE_VSL]),concat([SAMPLE_VCL]))
# plot_vcl_vsl_hist.m:47
    # Sort the dots so red is on top
    gg,II=sort(EE,'descend',nargout=2)
# plot_vcl_vsl_hist.m:50
    # Plot the histogram
    scatter(SAMPLE_VSL(II),SAMPLE_VCL(II),20,EE(II),'filled')