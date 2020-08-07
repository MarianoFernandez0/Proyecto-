function [h] = plotTrackHistory_4L(TrackRecord, T, startTime, endTime)

% How many tracks are there? 
trackList = unique(TrackRecord(:,1))';

% Plot each track 
for trk = trackList
    
    % Get data for this track
    dataIdx = find(TrackRecord(:,1) == trk);    
    posX = TrackRecord(dataIdx,5);
    posY = TrackRecord(dataIdx,6);
    measX = TrackRecord(dataIdx,19);
    measY = TrackRecord(dataIdx,20);
    time = TrackRecord(dataIdx,4).*T;
            
    % Speed 
    v = sqrt((diff(measX)./T).^2 + (diff(measY)./T).^2);

    % Plot moving tracks
    %%%%%%Cambio endTime a mano acá y startTime
    endTime = 50
    startTime = 0
    timeIdx = find(time >= startTime & time <= endTime);        
    %plot(measX(timeIdx), measY(timeIdx), 'k.'); 
    
    plot(measX(timeIdx), measY(timeIdx), 'k'); 
   
        
end
