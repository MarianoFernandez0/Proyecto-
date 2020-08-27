function [stats] = analyzeTrackRecord(TrackRecord, T, timePerSperm)

% /////////////////////////////////////////////////////////////////////////
%
%   Measure Sperm Motility Parameters from Track File
%   by Leonardo F. Urbano
%   April 5th, 2015
%
% /////////////////////////////////////////////////////////////////////////

% Time avg window (s)
avgTime = 1;

% Number of points in avg window
windowSize = ceil(avgTime/T);

% Analysis time per sperm  (sec)
% timePerSperm = 1;

% Clear the Records
sampleTRK = [];
sampleVCL = [];
sampleVSL = [];
sampleALH = [];
sampleLIN = [];
sampleVAP = [];
sampleWOB = [];
sampleSTR = [];
sampleMAD = [];

% Number of Confirmed Tracks to Analyze
trackList = unique(TrackRecord((TrackRecord(:,2) == 2),1));
numTracks = length(trackList);

% Draw wait bar
hWaitbar = waitbar(0, 'Sperm analysis in progress... ');

% Initialize the track count for the waitbar
trkCount = 0;

% Minimum number of track points to be analyzed 
numPoints = ceil(timePerSperm * windowSize);

% Analyze each track
for trk = trackList'
        
    % Set of measurements for this track
    dataIdx = find(TrackRecord(:,1) == trk);
    
    % Discard the first and last 5 points
    dataIdx = dataIdx(5:end-5);
    
    % If the track meets the minimum number of points, then perform
    % motility analysis     
    if ( length(dataIdx) > numPoints )
    
        % Take only numPoints worth of data
        dataIdx = dataIdx(1:numPoints);
        
        % Increase the track analysis count
        trkCount = trkCount + 1;
        
        % X-Y track points
        measX = TrackRecord(dataIdx,19);
        measY = TrackRecord(dataIdx,20);
        
        % Process Each Segment in the Path
        for kStep = 0:(numPoints - windowSize)
            
            VCL = []; VSL = []; LIN = []; ALH = [];
            VAP = []; WOB = []; MAD = []; STR = [];
            
            % Select 1-sec Track segment
            Zx = measX(kStep + 1: kStep + windowSize)';
            Zy = measY(kStep + 1: kStep + windowSize)';
            
            % Calculate VCL over the segment
            Vx = diff(Zx)/T;
            Vy = diff(Zy)/T;
            VCL = mean(sqrt(Vx.^2 + Vy.^2));
            
            % Calculate VSL over the segment
            DSLx = Zx(end) - Zx(1);
            DSLy = Zy(end) - Zy(1);
            VSL = sqrt(DSLx^2 + DSLy^2)/(windowSize * T);
            
            % Calculate LIN over the segment
            LIN = VSL/VCL;
            
            % Average Path (average 2 points before and ahead)
            Sx = []; Sy = [];
            for jjj = 3:(length(Zx)-2)
                Sx = [Sx mean(Zx(jjj-2:jjj+2))];
                Sy = [Sy mean(Zy(jjj-2:jjj+2))];
            end
            
            % Velocity average path
            VSx = diff(Sx)/T;
            VSy = diff(Sy)/T;
            VAP = mean(sqrt(VSx.^2 + VSy.^2));
            
            % Amplitude of lateral head displacement
            DLH = [Zx(3:end-2); Zy(3:end-2)] - [Sx; Sy];
            DEV = sqrt(DLH(1,:).^2 + DLH(2,:).^2);
            ALH = 2 * mean(DEV); % could also use
            % ALH = 2 * max(DEV);
            
            % Mean Angular Displacement
            MADi = [];
            for jjj = 2:length(Vx)
                mag1 = norm([Vx(jjj) Vy(jjj)]);
                uv1 = [Vx(jjj); Vy(jjj)]./mag1;
                mag2 = norm([Vx(jjj-1) Vy(jjj-1)]);
                uv2 = [Vx(jjj-1); Vy(jjj-1)]./mag2;
                MADi = [MADi acosd(dot(uv1,uv2))];
            end
            MAD = mean(MADi);
            if isreal(MAD)
                MAD = MAD;
            else
                MAD = 0;
            end
            
            
            % Wobble
            WOB = VAP/VCL;
            STR = VSL/VAP;
            
            % Add the data to the sample set
            if (VCL > 30) && (VCL <= 250) && (VSL > 0) && (VSL <= 150)
                
                sampleTRK = [sampleTRK trk];
                sampleVCL = [sampleVCL VCL];
                sampleVSL = [sampleVSL VSL];
                sampleLIN = [sampleLIN LIN];
                sampleALH = [sampleALH ALH];
                sampleVAP = [sampleVAP VAP];
                sampleWOB = [sampleWOB WOB];
                sampleSTR = [sampleSTR STR];
                sampleMAD = [sampleMAD MAD];
                
            end
            
        end
        
    end
    
    % Update the waitbar
    waitbar(trkCount/numTracks, hWaitbar);
    
end

% Store stats
stats.sampleTRK = sampleTRK;
stats.sampleVCL = sampleVCL;
stats.sampleVSL = sampleVSL;
stats.sampleLIN = sampleLIN;
stats.sampleALH = sampleALH;
stats.sampleVAP = sampleVAP;
stats.sampleWOB = sampleWOB;
stats.sampleSTR = sampleSTR;
stats.sampleMAD = sampleMAD;
stats.trackCount = trkCount;

close(hWaitbar);

