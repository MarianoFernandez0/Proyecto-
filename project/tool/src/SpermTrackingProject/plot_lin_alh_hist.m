function [h] = plot_lin_alh_hist(SAMPLE_ALH, SAMPLE_LIN)

% Plots the 2D histogram of ALH and LIN
% SAMPLE_ALH = [SAMPLE_ALH -0.01 20.01];
% SAMPLE_LIN = [SAMPLE_LIN -0.01 1.01];


% figure; hold on; grid on; 
axis([0 20 0 1]);
set(gcf, 'PaperPositionMode', 'Auto')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Label the histogram 
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% title(['Kinematic Properties of the Sample'], 'FontSize', 12, ...
%     'FontName', 'Arial', 'FontWeight', 'Bold');
xlabel('ALH (\mum)', 'FontSize', 18, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');
ylabel('LIN = VCL/VSL', 'FontSize', 18, ...
    'FontName', 'Arial', 'FontWeight', 'Bold');
set(gca, 'FontSize', 18, ...
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
%   Normalize the data
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

[NN, CC] = hist3([SAMPLE_ALH' SAMPLE_LIN'], [25 25]);
DD = NN./(max(max(NN)));

% Interpolate the data matrix
EE = interp2(CC{1}, CC{2}, DD', SAMPLE_ALH, SAMPLE_LIN);

% Sort the data so red is on top
[gg, II] = sort(EE, 'descend');

% Draw the scatter plot
scatter(SAMPLE_ALH(II), SAMPLE_LIN(II), 20, EE(II), 'filled');




