function [] = play_segmentation(filename)
% Plays the segmented video file results

% Load the segmentation video mat file
load(filename)

% Play Detection Movie
figure(1); hold on; grid on;
axis([0 640 0 480]);

% Number of frames
[video_length, ~] = size(fieldnames(data));

for k = 1:video_length
    
    frame = sprintf('frame%d', k);
    
    % Plot the Points
    h = plot(data.(frame).x, data.(frame).y, 'r+', 'MarkerSize', 12);
    
    % Pause for Animation
    pause(0.05);
    
    % Delete the Points
    if (k < video_length)
        
        delete([h(1)]);
        
    end
    
end
