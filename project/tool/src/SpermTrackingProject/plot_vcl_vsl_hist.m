function [h] = plot_vcl_vsl_hist(SAMPLE_VSL, SAMPLE_VCL)

% Plots the 2D histogram of VSL and VSL
% SAMPLE_VCL = [SAMPLE_VCL -0.01 250.01];
% SAMPLE_VSL = [SAMPLE_VSL -0.01 150.01];

% h = figure; hold on; grid on; 
axis([0 150 0 250]);
% set(gcf, 'PaperPositionMode', 'Auto')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Label the histogram 
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% title(['Kinematic Properties of the Sample'], 'FontSize', 12, ...
%     'FontName', 'Arial', 'FontWeight', 'Bold');
xlabel('VSL (\mum/s)', 'FontSize', 18, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');
ylabel('VCL (\mum/s)', 'FontSize', 18, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');
set(gca, 'FontSize', 14, ...
    'FontName', 'Arial', 'FontWeight', 'Bold', 'Box', 'On')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Draw colorbar
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
colorbar_handle = colorbar;
ylabel(colorbar_handle, ...
    'Relative Density of Data Points', 'FontSize', 18, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');
caxis([0 1]);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%  Normalize the data
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

[NN, CC] = hist3([SAMPLE_VSL' SAMPLE_VCL'], [25 25]);
DD = NN./(max(max(NN)));

% Interpolate the data matrix
EE = interp2(CC{1}, CC{2}, DD', [SAMPLE_VSL], [SAMPLE_VCL]);

% Sort the dots so red is on top
[gg, II] = sort(EE, 'descend');

% Plot the histogram
scatter(SAMPLE_VSL(II), SAMPLE_VCL(II), 20, EE(II), 'filled');




