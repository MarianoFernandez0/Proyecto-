function [h] = plot_lin_alh_hist(SAMPLE_ALH, SAMPLE_LIN)

% Plots the 2D histogram of ALH and LIN

h = figure; 
hold on; grid on; axis([0 20 0 1]);



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Label the histogram 
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

title(['Kinematic Properties of the Sample'], 'FontSize', 12, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');

xlabel('Amplitude Lateral Head Displacement ALH (\mum)', 'FontSize', 12, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');

ylabel('Linearity VSL /VCL (unitless)', 'FontSize', 12, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');

set(gca, 'FontSize', 12, ...
    'FontName', 'Arial', 'FontWeight', 'Bold')



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Draw the colorbar
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Draw the colorbar
colorbar_handle = colorbar;

% Label the colorbar
ylabel(colorbar_handle, 'Relative Density of Data Points', 'FontSize', 12, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');

% Set the colorbar limits
caxis([0 1]);

% Generate histogram data the Scatter Plot
[NN, CC] = hist3([SAMPLE_ALH' SAMPLE_LIN'], [25 25]);

% Normalize the colors
DD = NN./(max(max(NN)));

% Interpolate the data matrix
EE = interp2(CC{1}, CC{2}, DD', SAMPLE_ALH, SAMPLE_LIN);

% Sort the data so red is on top
[gg, II] = sort(EE, 'descend');

% Draw the scatter plot
plotHandles{end+1} = scatter(SAMPLE_ALH(II), SAMPLE_LIN(II), ...
    20, EE(II), 'filled');