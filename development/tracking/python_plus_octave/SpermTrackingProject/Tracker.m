% /////////////////////////////////////////////////////////////////////// %
%
%   JPDA Multi-Target Tracking Algorithm
%
%   By  Leonardo F. Urbano
%
%   April 11th, 2015
%
%
% /////////////////////////////////////////////////////////////////////// %

function Tracker(dataFile, videoFile, videoFileOut, csvTracks, reformat_dataFile, numFrames, fps, px2um, ROIx, ROIy, mttAlgorithm, PG, PD, gv, plotResults, saveMovie, snapShot, plotTrackResults, analyzeMotility)
	pkg load statistics
	pkg load video
	pkg load image
	pkg load io
	%Comento esto para probar                      --------MODIFICADO
	%s = RandStream('mt19937ar','Seed',1);        %--------MODIFICADO
	%RandStream.setGlobalStream(s)                %--------MODIFICADO

	% Load Data File	%-------------------------------------------------------------------------------------------------------------------------------------------------------------
	if reformat_dataFile
		zTotal_aux = csvread(dataFile);
		zTotal = zTotal_aux(:,[3 2 4])';
		
	else
		zTotal = csvread(dataFile);
	end
	%-------------------------------------------------------------------------------------------------------------------------------------------------------------

	% Load Video File
	video = VideoReader(videoFile);

	% Scale Factor (px2micron)
	um2px = 1/px2um;

	% Initalize Filter
	T = 1/10;   % (sec) sample period

	% Dynamical System & Measurement Equation
	F = [1 0 T 0; 0 1 0 T; 0 0 1 0; 0 0 0 1];
	G = [T^2/2 0; 0 T^2/2; T 0; 0 T];
	H = [1 0 0 0; 0 1 0 0];

	% Noise Covariance Matrix
	N = 2^2 * eye(2,2);

	% CWNA Process Noise
	qMat = [T^3/3 0 T^2/2 0; 0 T^3/3 0 T^2/2; T^2/2 0 T 0; 0 T^2/2 0 T];
	%deltaV = 20;                             %--------MODIFICADO
	deltaV = 80;                              %--------MODIFICADO
	qIntensity = deltaV^2/T;                  %--------MODIFICADO
	qIntensity = 20;                  	   %--------MODIFICADO
	Q0 = qMat * qIntensity;

	% Initial covariance matrix
	P0 = [N N./T; N./T 2*N./T^2];  		%MODIFICADO Este anda mejor
	%P0 = Q0; % [2^2 0 0 0; 0 2^2 0 0; 0 0 1 0; 0 0 0 1];

	% Initial residual covariance matrix
	S0 = H * P0 * H' + N;



	% Expected number of measurements due to clutter per unit area of the
	% surveillance space per scan of data
	lam_f = 1e-8;

	% Expected number of measurements from new targets per unit area of the
	% surveillance space per scan of data
	lam_n = 1e-3;

	% Track Management Thresholds
	initialScore = log(lam_n/lam_f);
	Pdelete = 1e-6;
	Pconfirm = 1e-4;
	threshDelete = log(Pdelete / (1 - Pconfirm));
	threshConfirm = log((1-Pdelete)/Pconfirm) + initialScore;

	% Position Gate (um)
	gx = chi2inv(PG, 2);



	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Create Analysis Figures
	%
	% /////////////////////////////////////////////////////////////////////// %

	if (plotResults)
	    fHandles(1) = figure;
	    set(gca, 'Box', 'On'); axis equal;
	    axis([0 ROIx 0 ROIy]);  hold on; grid on;
	end

	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Main Loop
	%
	% /////////////////////////////////////////////////////////////////////// %

	% Clear the Track File
	TrackFile = [];

	% Clear the Track Record
	TrackRecord = [];

	% Load the waitbar
	hWaitbar = waitbar(0, 'Processing ...');


	% Process Each Frame
	for k = 1:numFrames
	    
	    m = 0;  % Initialize Number of Measurements at Frame k
	    n = 0;  % Initialize Number of Active Tracks at Frame k
	    
	    % Set of Measurements at Frame k assuming PD
	    Z = [];
	    Z = zTotal(1:2, (zTotal(3,:) == k)) .* px2um;
	    
	    % Number of Measurements at Frame k
	    if ~isempty(Z)
		[~, m] = size(Z);
		% Mark all Measurements as Un-used
		Z(3,:) = 0;
	    end
	    
	    % List of Active Tracks
	    if ~isempty(TrackFile)
		t_idx = find(TrackFile(:,1) > 0);
		n = length(t_idx);
	    end
	    
	    
	    % /////////////////////////////////////////////////////////////////// %
	    %
	    %   Tracking Loop
	    %
	    % /////////////////////////////////////////////////////////////////// %
	    if ~isempty(TrackFile)
		
		% Indices of all Measurements at Frame k
		m_idx = 1:m;
		
		% Clear Temporary Lists
		Xu = []; Pu = [];
		Xp = []; Pp = []; Sp = []; Zp = [];
		d = []; f = [];
		
		% Predict the State and Covariance of each Active Track
		for t = 1:length(t_idx)
		    
		    % Track Number in the TrackFile
		    trk = t_idx(t);           
		    
		    % Save the sigmaN to the trackfile
		    sigmaN = TrackFile(trk, 25);
		    
		    % Measurement noise covariance matrix
		    Nk(:,:,t) = sigmaN^2 * eye(2,2);
		                           
		    % Load Track t State and Covariance
		    Xu(:,t) = TrackFile(trk, 4:7)';
		    Pu(:,:,t) = [TrackFile(trk,8:11);  TrackFile(trk,[9 12:14]); TrackFile(trk,[10 13 15:16]); TrackFile(trk, [11 14 16 17])];
		    oldQhat = [TrackFile(trk,26:29); TrackFile(trk,[27 30:32]); TrackFile(trk,[28 31 33:34]); TrackFile(trk,[29 32 34 35])];             
		    
		    % Predicted State
		    Xp(:,t) = F * Xu(:,end);
		    
		    % Differene between state and predicted state
		    deltaX = Xp(:,t) - Xu(:,end);
		    
		    % Adaptive process noise
		    c1 = 0.3;
		    c2 = 0.5;
		    c3 = 0.2;
		    newQhat = c1 * oldQhat + c2 * deltaX * deltaX' + c3 * Q0;
		            
		    TrackFile(end,26) = newQhat(1,1);  % Updated Q
		    TrackFile(end,27) = newQhat(1,2);  % Updated Q
		    TrackFile(end,28) = newQhat(1,3);  % Updated Q
		    TrackFile(end,29) = newQhat(1,4);  % Updated Q
		    TrackFile(end,30) = newQhat(2,2);  % Updated Q
		    TrackFile(end,31) = newQhat(2,3);  % Updated Q
		    TrackFile(end,32) = newQhat(2,4);  % Updated Q
		    TrackFile(end,33) = newQhat(3,3);  % Updated Q
		    TrackFile(end,34) = newQhat(3,4);  % Updated Q
		    TrackFile(end,35) = newQhat(4,4);  % Updated Q
		                
		    % Predicted Covariance and Residual Covariance                                    
		    Pp(:,:,t) = F * Pu(:,:,end) * F' + newQhat;
		    Sp(:,:,t) = H * Pp(:,:,end) * H' + Nk(:,:,t);
		    
		    % sqrt of determinant of 2pi Sp
		    sqrtDet2piSp = sqrt(det(2*pi*Sp(:,:,t)));
		    
		    % Predicted Measurement
		    Zp(:,t) = H * Xp(:,t);
		    
		    % Calculate the Distance Matrix Entries for this Track
		    for j = 1:m
		        
		        % Measurement Residual
		        v_jt = Z(1:2,j) - Zp(:,t);
		        
		        % Normalized Statistical Distance
		        d_jt = v_jt' / Sp(:,:,t) * v_jt;
		        
		        % Measurement Gating
		        if (d_jt <= gx) && (norm(v_jt)/T <= gv)
		            
		            % Distance between Track t and Measurement j
		            d(j,t) = d_jt;
		            
		            % Mark this Measurement as Used
		            Z(3,j) = 1;
		            
		        else
		            
		            % No Measuremnt in Validation Gates
		            d(j,t) = Inf;
		            
		        end
		        
		        % Gaussian pdf
		        f(j,t) =  exp(-0.5 * d(j,t)) / sqrtDet2piSp;
		        
		    end % // Distance Matrix Loop
		    
		end % // Track Loop
		
		
		
		% /////////////////////////////////////////////////////////// %
		%
		%   Identify Track Clusters
		%
		% /////////////////////////////////////////////////////////// %
		
		% Association Matrix
		A = ceil(f);
		[mA,nA] = size(A);
		
		
		% If A is not empty
		if ~isempty(A)
		    
		    % Identify Track Clusters (tracks gated by measurements)
		    clusters = identifyClusters(A);
		    
		    % Number of Clusters
		    numClusters = unique(clusters(:,1))';
		    
		else
		    
		    % If there are no measurements at all, then each track is
		    % its own cluster
		    clusters = [(1:length(t_idx))' zeros(length(t_idx),1) (1:length(t_idx))'];
		    numClusters = length(t_idx);
		    
		end
		
		
		
		% /////////////////////////////////////////////////////////// %
		%
		%   Proceses Each Track Cluster
		%
		% /////////////////////////////////////////////////////////// %
		
		% Process Each Cluster
		for c_idx = numClusters
		    
		    jj_idx = [];
		    tt_idx = [];
		    
		    % Indices of Measurements in Cluster c_idx
		    jj_idx = unique(clusters(find(clusters(:,1) == c_idx), 2))';
		    
		    % Inidices of Tracks in Cluster c_idx
		    tt_idx = unique(clusters(find(clusters(:,1) == c_idx), 3))';
		    
		    
		    % /////////////////////////////////////////////////////// %
		    %
		    %   Data Association
		    %
		    %//////////////////////////////////////////////////////// %
		    
		    beta = 0;
		    LR = 0;
		    
		    % If at least one measurment validates track(s)
		    if (jj_idx > 0)
		        
		        % Likelihood Ratio Matrix
		        LR = (lam_f^-1) .* f(jj_idx, tt_idx) .* PD;
		        
		        if (mttAlgorithm == 1)
		            
		            % Nearest-neighbor
		            beta = nnAssociation(LR);
		            
		        elseif (mttAlgorithm == 2)
		            
		            % Global Nearest Neighbor
		            beta = munkres(-log(LR));
		            
		        elseif (mttAlgorithm == 3)
		            
		            % PDAF
		            beta = pdafAssociation(LR, PD, PG);
		            
		        elseif (mttAlgorithm == 4);
		            
		            % JPDAF
		            beta = jpdafAssociation(LR, PD, PG);
		            
		        elseif (mttAlgorithm == 5);
		            
		            % ENN-JPDA
		            beta = munkres(-log(jpdafAssociation(LR, PD, PG)));
		                                
		        elseif (mttAlgorithm == 6)
		            
		            % Iterated multi-assignment
		            beta = imaAlgorithm(LR, PD, PG);
		            
		        end
		        
		    end
		    
		    
		    % /////////////////////////////////////////////////////// %
		    %
		    %   Track Update
		    %
		    % /////////////////////////////////////////////////////// %
		    
		    % For Each Track tt in the Cluster
		    for tt = 1:length(tt_idx)
		        
		        % Prob-weighted Combined Innovation
		        V_t = [0; 0];
		        
		        % Combined Distance
		        D_t = 0;
		        
		        v_jt = [0; 0];
		                        
		        atLeastOneMeasurement = 0;
		        
		        % If at least one measurement validates the track
		        if (jj_idx > 0)
		                       
		            atLeastOneMeasurement = 1;
		            
		            % For Each Measurement in the Cluster
		            for jj = 1:length(jj_idx)
		                
		                % Resudial Between Meas jj and Track tt
		                v_jt = Z(1:2,jj_idx(jj)) - Zp(:,tt_idx(tt));
		                
		                % Prob-weighted Combined Residual
		                V_t = V_t + beta(jj,tt) * v_jt;
		                
		                % Spread-of-the-means
		                D_t = D_t + beta(jj,tt) * (v_jt * v_jt');
		                
		            end
		            
		        end
		        
		        % Probability that None of the Measurements is Correct
		        beta0 = (1 - PD * PG) / (1 - PD * PG + sum(LR(:,tt)));
		        
		        % Filter Gain
		        K = Pp(:,:,tt_idx(tt)) * H' / Sp(:,:,tt_idx(tt));
		        
		        % Intermediate Matrix L
		        L = eye(4,4) - K * H;
		        
		        % P star
		        P_star = L * Pp(:,:,tt_idx(tt)) * L' + K * Nk(:,:,tt_idx(tt)) * K';
		        
		        % P zero
		        P_zero = beta0 * Pp(:,:,tt_idx(tt)) + (1 - beta0) * P_star;
		        
		        % P delta - increment for uncertain association
		        P_delta = K * (D_t - V_t * V_t') * K';
		        
		        % Updated covariance (multiply P_delta by zero to suppress)
		        Pu = P_zero + P_delta;
		        
		        % State update
		        Xu = Xp(:,tt_idx(tt)) + K * V_t;
		        
		        % Associated Measurement
		        Zm = H * Xp(:,tt_idx(tt)) + V_t;
		        
		        % Index of this track in the TrackFile
		        trk = t_idx(tt_idx(tt));
		                        
		        % Track Score  logic                                 
		        if (norm(V_t) == 0) && (atLeastOneMeasurement == 0)
		            
		            % If no measurement updated this track
		            deltaL = log(1 - PD);
		            
		            % If this track was just created, and didn't get a
		            % measurement, then delete it
		            if (TrackFile(trk,24) == 0)
		                TrackFile(trk,1) = 0;
		            end
		            
		        else
		            
		            % If a measurement updated the track
		            deltaL = log( lam_f^-1 * PD * ...
		                (2 * pi * det(Sp(:,:,tt_idx(tt))))^(-0.5) ...
		                - 0.5 * V_t' / Sp(:,:,tt_idx(tt)) * V_t );
		            
		            % Update the track update counter
		            TrackFile(trk,24) = TrackFile(trk,24) + 1;
		            
		        end
		        
		        % Update Track File                
		        TrackFile(trk, 2) = TrackFile(trk, 2) + deltaL;   % Track Score
		        TrackFile(trk, 3) = k;             % Frame Number
		        TrackFile(trk, 4) = Xu(1);         % Updated X Position at Frame k
		        TrackFile(trk, 5) = Xu(2);         % Updated Y Position at Frame k
		        TrackFile(trk, 6) = Xu(3);         % Updated X Velocity at Frame k
		        TrackFile(trk, 7) = Xu(4);         % Updated Y Velocity at Frame k
		        TrackFile(trk, 8) = Pu(1,1);       % Updated Track Covariance (1,1)
		        TrackFile(trk, 9) = Pu(1,2);       % Updated Track Covariance (1,2)
		        TrackFile(trk,10) = Pu(1,3);       % Updated Track Covariance (1,3)
		        TrackFile(trk,11) = Pu(1,4);       % Updated Track Covariance (1,4)
		        TrackFile(trk,12) = Pu(2,2);       % Updated Track Covariance (2,2)
		        TrackFile(trk,13) = Pu(2,3);       % Updated Track Covariance (2,3)
		        TrackFile(trk,14) = Pu(2,4);       % Updated Track Covariance (2,4)
		        TrackFile(trk,15) = Pu(3,3);       % Updated Track Covariance (3,3)
		        TrackFile(trk,16) = Pu(3,4);       % Updated Track Covariance (3,4)
		        TrackFile(trk,17) = Pu(4,4);       % Updated Track Covariance (4,4)
		        TrackFile(trk,18) = Zm(1);        % X Associated Position Measurement
		        TrackFile(trk,19) = Zm(2);        % Y Associated Position Measurement
		        TrackFile(trk,20) = max(TrackFile(trk,2)-deltaL, TrackFile(trk,20));
		        TrackFile(trk,21) = Sp(1,1,tt_idx(tt));       % Residual pos covariance xx
		        TrackFile(trk,22) = Sp(1,2,tt_idx(tt));       % Residual pos covariance xy
		        TrackFile(trk,23) = Sp(2,2,tt_idx(tt));       % Residual pos covariance yy                                
		        
		        
		        % Update the Track Record for Confirmed Tracks Only
		        if (TrackFile(trk, 1) > 0)
		            TrackRecord(end+1,:) = [trk TrackFile(trk, :)];
		        end
		        
		    end
		    
		end
		
		
		
		% /////////////////////////////////////////////////////////////// %
		%
		%   Plot the Problem
		%
		% /////////////////////////////////////////////////////////////// %
		
		if (plotResults == 1)
		    
		    pHandles = [];
		    
		    % Display the video frame
		    %currFrame = rgb2gray(read(video, k));             %MODIFICADO(matlab->octave)
		    currFrame = rgb2gray(readFrame(video, k));         %MODIFICADO(matlab->octave)            ˙
		    pHandles(end+1) = imshow(currFrame); hold on;
		    
		    % Plot the Measurements at Frame k
		    pHandles(end+1) = plot(Z(1,:) .* um2px, Z(2,:) .* um2px, 'r+');
		    pHandles(end+1) = plot(Xp(1,:) .* um2px, Xp(2,:) .* um2px, 'b.');
		    
		    % Set of Confirmed Tracks
		    ct_idx = find(TrackFile(:,1) > 1);
		    
		    % Plot the Predicted Tracks and Track Gates at Frame k
		    for t = 1:length(ct_idx)
		        
		        trk = ct_idx(t);
		        %pHandles(end+1) = plotEllipse(Zp(:,t)' .* um2px, gx * Sp(:,:,t) .* um2px^2);    %MODIFICADO
		        pHandles(end+1) = plot_gaussian_ellipsoid(Zp(:,t)' .* um2px, gx * Sp(:,:,t) .* um2px^2);    %MODIFICADO
		        set(pHandles(end), 'Color', 'r');
		        
		        pHandles(end+1) = text(Zp(1,t) .* um2px + 0.025 * ROIx, Zp(2,t) .* um2px + 0.025 * ROIy, ...
		            num2str(t_idx(t)), 'FontSize', 12, 'FontWeight', 'bold', ...
		            'FontName', 'Arial', 'Color', 'g');         %MODIFICADO k->g
		    end
		    
		    
		    pHandles(end+1) = plot(Z(1,:).* um2px, Z(2,:).* um2px, 'r+');
		    
		    pause(2);                                       %MODIFICADO
		    %pause(0.001);                                  %MODIFICADO
		    
		    if (k < numFrames)
		        delete(pHandles(:));
		    end
		    
		end
		
		
	    end
	    

	    
	    
	    % /////////////////////////////////////////////////////////////////// %
	    %
	    %   Track Promotion / Deletion
	    %
	    % /////////////////////////////////////////////////////////////////// %
	    if ~isempty(TrackFile)
		
		t_idx = find(TrackFile(:,1) > 0);
		
		% Check the Score of Each Track
		for t = 1:length(t_idx)
		    
		    trk = t_idx(t);
		    
		    % Promote the Track?
		    if (TrackFile(trk,2) > threshConfirm)
		        TrackFile(trk,1) = 2;
		    end
		    
		    % Change from Maximum Track Score
		    scoreDelta = TrackFile(trk,2) - TrackFile(trk,20);
		    
		    % Delete the Track if the scoreDelta Dropped too Much
		    if (scoreDelta <= threshDelete)
		        TrackFile(trk,1) = 0;
		    end
		    
		end
		
	    end
		
	    
	    % /////////////////////////////////////////////////////////////// %
	    %
	    %   Delete Duplicate Tracks
	    %
	    % /////////////////////////////////////////////////////////////// %
	    if ~isempty(TrackFile)
		
		% Set of indices of confirmed tracks
		t_idx = find(TrackFile(:,1) > 1);
		
		if ~isempty(t_idx) && (length(t_idx) > 1)
		    
		    XU = [TrackFile(t_idx,4) TrackFile(t_idx,5)];
		    
		    % Calculate distance between XU(jj) and XU(tt)
		    for jj = 1:length(XU)
		        
		        DD_idx = [t_idx(jj)];
		        
		        for tt = 1:length(XU)
		            
		            DD = sqrt((XU(jj,1) - XU(tt,1))^2 - (XU(jj,2) - XU(tt,2))^2);
		            
		            if (DD > 0) && (DD < 0.01)  % used to be 0.3
		                
		                DD_idx = [DD_idx t_idx(tt)];
		                
		            end
		            
		        end
		        
		        % Which one of the set of redundant tracks has the
		        % highest track score?
		        
		        DD_idx_max = find(TrackFile(DD_idx, 2) == max(TrackFile(DD_idx, 2)));
		        
		        DD_max = DD_idx(DD_idx_max);
		        
		        TracksToDelete = setdiff(DD_idx, DD_max);
		        
		        for del_idx = 1:length(TracksToDelete)
		            
		            TrackFile(TracksToDelete(del_idx),1) = 0;
		            
		        end
		        
		    end
		    
		end
		
	    end
	    

	    % /////////////////////////////////////////////////////////////////// %
	    %
	    %   Initiate Tracks on Un-used Measurements
	    %
	    % /////////////////////////////////////////////////////////////////// %
	    if ~isempty(Z)
		for j = find(Z(3,:) == 0)
		    TrackFile(end+1,1) = 1;             % Track Rank
		    % 1 = Tentative
		    % 2 = Confirmed
		    % 0 = Deleted
		    TrackFile(end,  2) = initialScore;  % Track Score
		    TrackFile(end,  3) = k;             % Frame Number
		    TrackFile(end,  4) = Z(1,j);        % Updated X Position at Frame k
		    TrackFile(end,  5) = Z(2,j);        % Updated Y Position at Frame k
		    TrackFile(end,  6) = 0;             % Updated X Velocity at Frame k
		    TrackFile(end,  7) = 0;             % Updated Y Velocity at Frame k
		    TrackFile(end,  8) = P0(1,1);       % Updated Track Covariance (1,1)
		    TrackFile(end,  9) = P0(1,2);       % Updated Track Covariance (1,2)
		    TrackFile(end, 10) = P0(1,3);       % Updated Track Covariance (1,3)
		    TrackFile(end, 11) = P0(1,4);       % Updated Track Covariance (1,4)
		    TrackFile(end, 12) = P0(2,2);       % Updated Track Covariance (2,2)
		    TrackFile(end, 13) = P0(2,3);       % Updated Track Covariance (2,3)
		    TrackFile(end, 14) = P0(2,4);       % Updated Track Covariance (2,4)
		    TrackFile(end, 15) = P0(3,3);       % Updated Track Covariance (3,3)
		    TrackFile(end, 16) = P0(3,4);       % Updated Track Covariance (3,4)
		    TrackFile(end, 17) = P0(4,4);       % Updated Track Covariance (4,4)
		    TrackFile(end, 18) = Z(1,j);        % X Associated Position Measurement
		    TrackFile(end, 19) = Z(2,j);        % Y Associated Position Measurement
		    TrackFile(end, 20) = initialScore;  % Maximum Track Score
		    TrackFile(end,21) = S0(1,1);       % Residual pos covariance xx
		    TrackFile(end,22) = S0(1,2);       % Residual pos covariance xy
		    TrackFile(end,23) = S0(2,2);       % Residual pos covariance yy
		    TrackFile(end,24) = 0;             % Updated track counter
		    TrackFile(end,25) = sqrt(N(1,1));  % Initial sigmaN
		    
		    TrackFile(end,26) = Q0(1,1);  % Initial Q
		    TrackFile(end,27) = Q0(1,2);  % Initial Q
		    TrackFile(end,28) = Q0(1,3);  % Initial Q
		    TrackFile(end,29) = Q0(1,4);  % Initial Q
		    TrackFile(end,30) = Q0(2,2);  % Initial Q
		    TrackFile(end,31) = Q0(2,3);  % Initial Q
		    TrackFile(end,32) = Q0(2,4);  % Initial Q
		    TrackFile(end,33) = Q0(3,3);  % Initial Q
		    TrackFile(end,34) = Q0(3,4);  % Initial Q
		    TrackFile(end,35) = Q0(4,4);  % Initial Q
		end
	    end
		
	    % Track Record Defintion
	    % TrackRecord(:,1)  Track Number
	    % TrackRecord(:,2)  Track Rank (0 = deleted, 1 = tentative, 2 = confirmed)
	    % TrackRecord(:,3)  Track Score
	    % TrackRecord(:,4)  Frame Number
	    % TrackRecord(:,5)  Estimated X position at frame k
	    % TrackRecord(:,6)  Estimated Y position at frame k
	    % TrackRecord(:,19)  Measured Y position at frame k
	    % TrackRecord(:,20)  Measured Y position at frame k
	    
	    
	    % Update the waitbar
	    waitbar(k/numFrames, hWaitbar);
	end % // Main Loop


	%------------------------------------------------------------------------------
	% guardo los resultados en un csv
	tracks_csv = [];

	%tracks_csv(:,1) = TrackRecord(:,1); 
	%tracks_csv(:,2) = TrackRecord(:,5);
	%tracks_csv(:,3) = TrackRecord(:,6);
	%tracks_csv(:,2) = TrackRecord(:,19);
	%tracks_csv(:,3) = TrackRecord(:,20);
	%tracks_csv(:,1) = TrackRecord(:,4);


	%1 19 20 4 = id_trk, x, y, frame
	csvwrite(csvTracks, 'a')
	csvwrite(csvTracks, TrackRecord(:,[1 19 20 4]))

	%------------------------------------------------------------------------------
	% Close the waitbar
	close(hWaitbar);







	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Create Movie File
	%
	% /////////////////////////////////////////////////////////////////////// %

	%saveMovie = 1;

	if (saveMovie)

	    figure; set(gca, 'Box', 'On');
	    %iptsetpref('ImshowBorder', 'tight')           %MODIFICADO (no existe en octave)
	    
	    % Open the Movie File
	    videoFileOut
	    vidObj = VideoWriter(videoFileOut);
	    %set(vidObj, 'Quality', 100);			%MODIFICADO
	    %set(vidObj, 'FrameRate', 1/T);			%MODIFICADO 
	    open(vidObj)					%MODIFICADO 

	    % Length of Trail History (in frames)
	    trailLength = (1 * 15);
	    
	    % Loop over each Frame
	    for k = 1:numFrames
		
		% Display the video frame
		%imshow(imcomplement(rgb2gray(read(video, k))));          %MODIFICADO (matlab->octave)
		imshow(imcomplement(rgb2gray(readFrame(video, k))));      %MODIFICADO (matlab->octave)
		% imshow(rgb2gray(read(video, k)));
		hold on;
		set(gcf, 'Position', [255 90 955 715]);
		
		numTentativeTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
		
		numConfirmedTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
		
		
		% Find the Index to the data at time k for confirmed tracks
		timeIdx = find(TrackRecord(:,4) == k & ...
		    TrackRecord(:,2) == 2);
		
		% Find the tracks at time k
		trackIdx = unique(TrackRecord(timeIdx,1));
		

		
		% Plot each track up to time k
		for trk = trackIdx'
		    
		    % Get the indices to the data for this track up to time k
		    dataIdx = find(TrackRecord(:,1) == trk & ...
		        TrackRecord(:,4) <= k);
		    
		    % Predicted State, Covariance and Residual Covariance
		    SpMat(1,1) = TrackRecord(dataIdx(end), 22);
		    SpMat(1,2) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,1) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,2) = TrackRecord(dataIdx(end), 24);
		    
		    
		    % Plot the last trailLength# track points up to time k
		    if (length(dataIdx) <= trailLength)
		        posX = TrackRecord(dataIdx,5) .* um2px;
		        posY = TrackRecord(dataIdx,6) .* um2px;
		        measX = TrackRecord(dataIdx,19) .* um2px;
		        measY = TrackRecord(dataIdx,20) .* um2px;
		    else
		        posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
		        posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
		        measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
		        measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
		    end
		    
		    % Set of Measurements at Frame k assuming PD
		    Z = [];
		    Z = zTotal(1:2, (zTotal(3,:) == k));
		    plot(Z(1,:), Z(2,:), 'r+');
		    
		    % Plot the Tracks and Measurements up to time k
		    plot(posX , posY, 'b');
		    plot(posX(end) , posY(end), 'cs', 'MarkerSize', 20);
		    plot(measX, measY, 'g')
		    plot(measX, measY, 'g.', 'MarkerSize', 5);
		    plot(measX(end), measY(end), 'y+');
		    
		    % Plot the track gate
		    ellipHand = plot_gaussian_ellipsoid([posX(end); posY(end)], gx * SpMat .* um2px^2);  %MODIFICADO plotElipse->plot_gaussian_ellipsoid
		    set(ellipHand, 'Color', 'r');
		    
		    % Label the Track Number
		    text(posX(end)+5, posY(end)+5, num2str(trk), ...
		        'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold', 'Color', 'g');
		    
		end
		
		% Draw 100um scale bar
		%hRectangle = rectangle('Position', [20 460 100*um2px 5*px2um]);
		%set(hRectangle, 'FaceColor', 'w', 'EdgeColor', 'w');
		
		% Sign your name
		hText = text(440, 460, 'L. Urbano, et al (2015) Drexel University', 'FontSize', 12);
		set(hText, 'Color', 'w');
		
		% Wait (allow time to save)
		pause(0.01);
		
		% Save the frame to the movie file
		currFrame = getframe(gcf);
		if k>1
			writeVideo(vidObj, currFrame)           %MODIFICADO     
		end                         %MODIFICADO
		
		%Clear the frame
		if (k<numFrames)
		    clf
		end
		
		k
		
	    end
	    % Close the movie file
	    close(vidObj);
	    
	end






	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Tracking Snapshot
	%
	% /////////////////////////////////////////////////////////////////////// %

	%snapShot = 1;


	if (snapShot)
	    
	    figure; set(gca, 'Box', 'On');
	    %iptsetpref('ImshowBorder', 'tight')
	    
	    % Length of Trail History (in frames)
	    trailLength = (1 * 15);
	    
	    % Snapshot frame number
	    for k = 13 + [5 10 15 20 25 30]
		
		% Display the video frame
		%imshow(imcomplement(rgb2gray(read(video, k))));            %MODIFICADO
		imshow(imcomplement(rgb2gray(readFrame(video, k))));        %MODIFICADO
		
		% imshow(rgb2gray(read(video, k)));
		hold on;
		set(gcf, 'Position', [255 90 955 715]);
		
		numTentativeTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
		
		numConfirmedTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
		
		
		% Find the Index to the data at time k for confirmed tracks
		timeIdx = find(TrackRecord(:,4) == k & ...
		    TrackRecord(:,2) == 2);
		
		% Find the tracks at time k
		trackIdx = unique(TrackRecord(timeIdx,1));
		
		
		% Plot each track up to time k
		for trk = trackIdx'
		    
		    % Get the indices to the data for this track up to time k
		    dataIdx = find(TrackRecord(:,1) == trk & ...
		        TrackRecord(:,4) <= k);
		    
		    % Predicted State, Covariance and Residual Covariance
		    SpMat(1,1) = TrackRecord(dataIdx(end), 22);
		    SpMat(1,2) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,1) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,2) = TrackRecord(dataIdx(end), 24);
		    
		    
		    % Plot the last trailLength# track points up to time k
		    if (length(dataIdx) <= trailLength)
		        posX = TrackRecord(dataIdx,5) .* um2px;
		        posY = TrackRecord(dataIdx,6) .* um2px;
		        measX = TrackRecord(dataIdx,19) .* um2px;
		        measY = TrackRecord(dataIdx,20) .* um2px;
		    else
		        posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
		        posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
		        measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
		        measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
		    end
		    
		    % Plot the Tracks and Measurements up to time k
		    plot(posX , posY, 'b');
		    plot(measX, measY, 'g')
		    plot(measX, measY, 'g.', 'MarkerSize', 5);
		    plot(measX(end), measY(end), 'r+');
		    
		    % Plot the track gate
		    %ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);    %MODIFICADO
		    ellipHand = plot_gaussian_ellipsoid([posX(end); posY(end)], gx * SpMat .* um2px^2);
		    set(ellipHand, 'Color', 'r');
		    
		    
		    % Label the Track Number
		    text(posX(end)+5, posY(end)-5, num2str(trk), ...
		        'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold', 'Color', 'g');
		    
		    
		end
		
		% Zoom to frame
		set(gcf, 'Position', [48   607   201   198])
		% axis([330 330+150 25 25 + 150])
		axis([80 80+150 220 220+150])
		set(gcf, 'PaperPositionMode', 'Auto')
		set(gcf, 'renderer', 'painters');
		print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_NewSnapshotFrame', num2str(k), '.png'])
		
	    end
	    
	end









	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Track Results
	%
	% /////////////////////////////////////////////////////////////////////// %
	%plotTrackResults = 1;

	if (plotTrackResults)
	    
	    % Plot sperm tracks
	    figure; set(gcf, 'Position', [0 276 1440 461]);    
	    subplot(2,4, [1 2 5 6]); hold on;
	    
	    %plotTrackHistory_2L(TrackRecord, T, um2px)
	    plotTrackHistory_4L(TrackRecord, T, um2px, 0, numFrames) %lo cambié por 4L
	    set(gca, 'Box', 'On', 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'Bold')
	    axis equal; axis([0 ROIx 0 ROIy]/um2px);
	    
	    %xlabel(['X Position Coorindate (',char(181),'m)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    %ylabel(['Y Position Coorindate (',char(181),'m)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    
	    xlabel(['X Position Coorindate (''micro m)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    ylabel(['Y Position Coorindate (''microm)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    
	    title('Reconstructed Sperm Swimming Paths (Specimen A)', 'FontName', 'Arial', 'FontWeight', 'Bold', 'FontSize', 12)
	end


	% % /////////////////////////////////////////////////////////////////////// %
	% %
	% %   2D-t plot
	% %
	% % /////////////////////////////////////////////////////////////////////// %
	% plot2Dt = 1;
	% 
	% if (plot2Dt)
	%     
	%     figure; hold on; grid on; view(3);
	%     plot2DtTrackHistory(TrackRecord, T, um2px);
	%     
	% end



	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Sperm Motility Analysis
	%
	% /////////////////////////////////////////////////////////////////////// %
	%analyzeMotility = 1;

	if (analyzeMotility)
	    
	    % [stats] = analyzeTrackRecord_rev_3L(TrackRecord, T);
	    % [stats] = analyzeTrackRecord_rev_4L(TrackRecord, T);
	    %[stats] = analyzeTrackRecord_rev_5L(TrackRecord, T);
	    [stats] = analyzeTrackRecord(TrackRecord, T, 1); %MODIFICADO cambié a 7L
	    
	    TRKNUM = stats.sampleTRK;
	    VCL = stats.sampleVCL;
	    VSL = stats.sampleVSL;
	    ALH = stats.sampleALH;
	    VAP = stats.sampleVAP;
	    LIN = stats.sampleLIN;
	    WOB = stats.sampleWOB;
	    STR = stats.sampleSTR;
	    MAD = stats.sampleMAD;
	    
	    % VCL vs VSL
	    subplot(2,4,3);
	    scatterXY(VSL,VCL,25,1); axis([0 150 0 250]); grid on;
	    set(gca, 'Box', 'On', 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'Bold')
	    xlabel(['VSL (',char(181),'m/s)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    ylabel(['VCL (',char(181),'m/s)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    %axis square;
	    
	    % LIN vs ALH
	    subplot(2,4,4);
	    scatterXY(ALH,LIN,25,1); axis([0 20 0 1]); grid on;
	    set(gca, 'Box', 'On', 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'Bold')
	    xlabel(['ALH (',char(181),'m)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    ylabel('LIN = VSL/VCL', 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    %axis square;
	    
	    % WOB vs VCL
	    subplot(2,4,7);
	    scatterXY(VSL,WOB,25,1); axis([0 150 0 1]); grid on;
	    set(gca, 'Box', 'On', 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'Bold')
	    xlabel(['VSL (',char(181),'m/s)'], 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    ylabel('WOB = VAP/VCL', 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    %axis square;
	    
	    % LIN vs MAD
	    subplot(2,4,8);
	    scatterXY(MAD,LIN,25,1); axis([1 180 0 1]); grid on;
	    set(gca, 'Box', 'On', 'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'Bold')
	    xlabel('MAD (deg)', 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    ylabel('LIN = VSL/VCL', 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold');
	    %axis square;
	    
	    figuresize(20,6.5,'inches');
	    set(gcf, 'PaperPositionMode', 'Auto')
	    set(gcf, 'renderer', 'painters');
	    % print(gcf, '-dpng', '-r300', fullfile([videoFile, '_SampleAnalysis', num2str(mttAlgorithm), '.png']))
	    
	    
	    
	    
	    disp(['# Tracks Analyzed: ', num2str(stats.trkCount)])
	    disp(['VCL: ', num2str(mean(stats.sampleVCL)), ', ', num2str(std(stats.sampleVCL))])
	    disp(['VSL: ', num2str(mean(stats.sampleVSL)), ', ', num2str(std(stats.sampleVSL))])
	    disp(['LIN: ', num2str(mean(stats.sampleLIN)), ', ', num2str(std(stats.sampleLIN))])
	    disp(['ALH: ', num2str(mean(stats.sampleALH)), ', ', num2str(std(stats.sampleALH))])
	    disp(['VAP: ', num2str(mean(stats.sampleVAP)), ', ', num2str(std(stats.sampleVAP))])
	    disp(['WOB: ', num2str(mean(stats.sampleWOB)), ', ', num2str(std(stats.sampleWOB))])
	    disp(['STR: ', num2str(mean(stats.sampleSTR)), ', ', num2str(std(stats.sampleSTR))])
	    disp(['MAD: ', num2str(mean(stats.sampleMAD)), ', ', num2str(std(stats.sampleMAD))])
	    
	end


	% % /////////////////////////////////////////////////////////////////////// %
	% %
	% %   Analysis Time
	% %
	% % /////////////////////////////////////////////////////////////////////// %
	% analysisTime = 1;
	%
	% if (analysisTime)
	%
	%     for jjj = 1:5
	%
	%         timePerSperm = jjj;
	%
	%         [stats] = analyzeTrackRecord_rev_6L(TrackRecord, T, timePerSperm);
	%
	%         meanVCL(jjj) = mean(stats.sampleVCL);
	%         meanVSL(jjj) = mean(stats.sampleVSL);
	%         meanALH(jjj) = mean(stats.sampleALH);
	%         meanVAP(jjj) = mean(stats.sampleVAP);
	%         meanLIN(jjj) = mean(stats.sampleLIN);
	%         meanWOB(jjj) = mean(stats.sampleWOB);
	%         meanSTR(jjj) = mean(stats.sampleSTR);
	%         meanMAD(jjj) = mean(stats.sampleMAD);
	%
	%         stdVCL(jjj) = std(stats.sampleVCL);
	%         stdVSL(jjj) = std(stats.sampleVSL);
	%         stdALH(jjj) = std(stats.sampleALH);
	%         stdVAP(jjj) = std(stats.sampleVAP);
	%         stdLIN(jjj) = std(stats.sampleLIN);
	%         stdWOB(jjj) = std(stats.sampleWOB);
	%         stdSTR(jjj) = std(stats.sampleSTR);
	%         stdMAD(jjj) = std(stats.sampleMAD);
	%
	%     end
	%
	%     figure; hold on; grid on;
	%     plot(meanVCL, 'b')
	%     plot(meanVCL+stdVCL(1:5), 'r--')
	%     plot(meanVCL-stdVCL(1:5), 'r--')
	%     axis([1 5 0 100])
	% end







	% % /////////////////////////////////////////////////////////////////////// %
	% %
	% %   Tracking Snapshot Sequence 1
	% %
	% % /////////////////////////////////////////////////////////////////////// %
	%
	% snapShot = 1;
	%
	% if (snapShot)
	%
	%     figure; set(gca, 'Box', 'On');
	%     iptsetpref('ImshowBorder', 'tight')
	%
	%     % Length of Trail History (in frames)
	%     trailLength = (1 * 15);
	%
	%     % Snapshot frame number
	%     for k = ceil(42/T) + [1 2 3 4 5 6]*15
	%
	%         k
	%
	%         % Display the video frame
	%         % imshow(imcomplement(rgb2gray(read(video, k))));
	%         imshow(rgb2gray(read(video, k)));
	%
	%         % imshow(rgb2gray(read(video, k)));
	%         hold on;
	%         set(gcf, 'Position', [255 90 955 715]);
	%
	%         numTentativeTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
	%
	%         numConfirmedTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
	%
	%
	%         % Find the Index to the data at time k for confirmed tracks
	%         timeIdx = find(TrackRecord(:,4) == k & ...
	%             TrackRecord(:,2) == 2);
	%
	%         % Find the tracks at time k
	%         trackIdx = unique(TrackRecord(timeIdx,1));
	%
	%
	%         % Plot each track up to time k
	%         for trk = trackIdx'
	%
	%             % Get the indices to the data for this track up to time k
	%             dataIdx = find(TrackRecord(:,1) == trk & ...
	%                 TrackRecord(:,4) <= k);
	%
	%             % Predicted State, Covariance and Residual Covariance
	%             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
	%             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,2) = TrackRecord(dataIdx(end), 24);
	%
	%
	%             % Plot the last trailLength# track points up to time k
	%             if (length(dataIdx) <= trailLength)
	%                 posX = TrackRecord(dataIdx,5) .* um2px;
	%                 posY = TrackRecord(dataIdx,6) .* um2px;
	%                 measX = TrackRecord(dataIdx,19) .* um2px;
	%                 measY = TrackRecord(dataIdx,20) .* um2px;
	%             else
	%                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
	%                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
	%                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
	%                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
	%             end
	%
	%             % Plot the Tracks and Measurements up to time k
	%             plot(measX, measY, 'g')
	%             plot(measX, measY, 'g.', 'MarkerSize', 5);
	%             plot(posX , posY, 'b');
	%             plot(measX(end), measY(end), 'r+');
	%
	%             % Plot the track gate
	%             % ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
	%             % set(ellipHand, 'Color', 'c');
	%
	%             % Label the Track Number
	%             text(posX(end)+5, posY(end)-5, num2str(trk), ...
	%                 'FontName', 'Arial', 'FontSize', fontSize, 'FontWeight', 'Bold', 'Color', 'k');
	%
	%         end
	%
	%         % Zoom to frame
	%         set(gcf, 'Position', [48   607   201   198])
	%         %axis([330 330+150 25 25 + 150])
	%         axis([360 360+150 120 120+150])
	%
	%         set(gcf, 'PaperPositionMode', 'Auto')
	%         set(gcf, 'renderer', 'painters');
	%         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_NewSnapshotFrame', num2str(k), '.png'])
	%
	%      end
	%
	% end



	%
	%
	% % /////////////////////////////////////////////////////////////////////// %
	% %
	% %   Tracking Snapshot Sequence 2
	% %
	% % /////////////////////////////////////////////////////////////////////// %
	%
	% snapShot = 1;
	%
	% if (snapShot)
	%
	%     figure; set(gca, 'Box', 'On');
	%     iptsetpref('ImshowBorder', 'tight')
	%
	%     % Length of Trail History (in frames)
	%     trailLength = (1 * 15);
	%
	%     % Snapshot frame number
	%     for k = ceil(35/T) + [1 2 3 4 5 6]*7
	%
	%         k
	%
	%         % Display the video frame
	%         % imshow(imcomplement(rgb2gray(read(video, k))));
	%         imshow(rgb2gray(read(video, k)));
	%
	%         % imshow(rgb2gray(read(video, k)));
	%         hold on;
	%         set(gcf, 'Position', [255 90 955 715]);
	%
	%         numTentativeTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
	%
	%         numConfirmedTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
	%
	%
	%         % Find the Index to the data at time k for confirmed tracks
	%         timeIdx = find(TrackRecord(:,4) == k & ...
	%             TrackRecord(:,2) == 2);
	%
	%         % Find the tracks at time k
	%         trackIdx = unique(TrackRecord(timeIdx,1));
	%
	%
	%         % Plot each track up to time k
	%         for trk = trackIdx'
	%
	%             % Get the indices to the data for this track up to time k
	%             dataIdx = find(TrackRecord(:,1) == trk & ...
	%                 TrackRecord(:,4) <= k);
	%
	%             % Predicted State, Covariance and Residual Covariance
	%             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
	%             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,2) = TrackRecord(dataIdx(end), 24);
	%
	%
	%             % Plot the last trailLength# track points up to time k
	%             if (length(dataIdx) <= trailLength)
	%                 posX = TrackRecord(dataIdx,5) .* um2px;
	%                 posY = TrackRecord(dataIdx,6) .* um2px;
	%                 measX = TrackRecord(dataIdx,19) .* um2px;
	%                 measY = TrackRecord(dataIdx,20) .* um2px;
	%             else
	%                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
	%                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
	%                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
	%                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
	%             end
	%
	%             % Plot the Tracks and Measurements up to time k
	%             plot(measX, measY, 'g')
	%             plot(measX, measY, 'g.', 'MarkerSize', 5);
	%             plot(posX , posY, 'b');
	%             plot(measX(end), measY(end), 'r+');
	%
	%             % Plot the track gate
	%             % ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
	%             % set(ellipHand, 'Color', 'c');
	%
	%             % Label the Track Number
	%             text(posX(end)+5, posY(end)-5, num2str(trk), ...
	%                 'FontName', 'Arial', 'FontSize', fontSize, 'FontWeight', 'Bold', 'Color', 'k');
	%
	%         end
	%
	%         % Zoom to frame
	%         set(gcf, 'Position', [48   607   201   198])
	%         % axis([330 330+150 25 25 + 150])
	%         % axis([360 360+150 120 120+150])
	%         axis([180 180+150 108 108+150]);%
	%
	%         set(gcf, 'PaperPositionMode', 'Auto')
	%         set(gcf, 'renderer', 'painters');
	%         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_Seq2SnapshotFrame', num2str(k), '.png'])
	%
	%      end
	%
	% end



	%
	%
	%
	% % /////////////////////////////////////////////////////////////////////// %
	% %
	% %   Tracking Snapshot Sequence 3
	% %
	% % /////////////////////////////////////////////////////////////////////// %
	%
	% snapShot = 1;
	%
	% if (snapShot)
	%
	%     figure; set(gca, 'Box', 'On');
	%     iptsetpref('ImshowBorder', 'tight')
	%
	%     % Length of Trail History (in frames)
	%     trailLength = (1 * 15);
	%
	%     % Snapshot frame number
	%     for k = ceil(13/T) + [1 2 3 4 5 6]*7
	%
	%         k
	%
	%         % Display the video frame
	%         % imshow(imcomplement(rgb2gray(read(video, k))));
	%         imshow(rgb2gray(read(video, k)));
	%
	%         % imshow(rgb2gray(read(video, k)));
	%         hold on;
	%         set(gcf, 'Position', [255 90 955 715]);
	%
	%         numTentativeTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
	%
	%         numConfirmedTracks(k) = length(unique(TrackRecord(...
	%             find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
	%
	%
	%         % Find the Index to the data at time k for confirmed tracks
	%         timeIdx = find(TrackRecord(:,4) == k & ...
	%             TrackRecord(:,2) == 2);
	%
	%         % Find the tracks at time k
	%         trackIdx = unique(TrackRecord(timeIdx,1));
	%
	%
	%         % Plot each track up to time k
	%         for trk = trackIdx'
	%
	%             % Get the indices to the data for this track up to time k
	%             dataIdx = find(TrackRecord(:,1) == trk & ...
	%                 TrackRecord(:,4) <= k);
	%
	%             % Predicted State, Covariance and Residual Covariance
	%             SpMat(1,1) = TrackRecord(dataIdx(end), 22);
	%             SpMat(1,2) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,1) = TrackRecord(dataIdx(end), 23);
	%             SpMat(2,2) = TrackRecord(dataIdx(end), 24);
	%
	%
	%             % Plot the last trailLength# track points up to time k
	%             if (length(dataIdx) <= trailLength)
	%                 posX = TrackRecord(dataIdx,5) .* um2px;
	%                 posY = TrackRecord(dataIdx,6) .* um2px;
	%                 measX = TrackRecord(dataIdx,19) .* um2px;
	%                 measY = TrackRecord(dataIdx,20) .* um2px;
	%             else
	%                 posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
	%                 posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
	%                 measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
	%                 measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
	%             end
	%
	%             % Plot the Tracks and Measurements up to time k
	%             plot(measX, measY, 'g')
	%             plot(measX, measY, 'g.', 'MarkerSize', 5);
	%             plot(posX , posY, 'b');
	%             plot(measX(end), measY(end), 'r+');
	%
	%             % Plot the track gate
	%             % ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
	%             % set(ellipHand, 'Color', 'c');
	%
	%             % Label the Track Number
	%             text(posX(end)+5, posY(end)-5, num2str(trk), ...
	%                 'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold', 'Color', 'k');
	%
	%         end
	%
	%         % Zoom to frame
	%         set(gcf, 'Position', [48   607   201   198])
	%         % axis([330 330+150 25 25 + 150])
	%         % axis([360 360+150 120 120+150])
	%         % axis([180 180+150 108 108+150]);%
	%         % axis([25 25+150 275 275+150]);
	%         axis([380 380+150 260 260+150]);
	%
	%         set(gcf, 'PaperPositionMode', 'Auto')
	%         set(gcf, 'renderer', 'painters');
	%         print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_Seq3SnapshotFrame', num2str(k), '.png'])
	%
	%      end
	%
	% end








	% /////////////////////////////////////////////////////////////////////// %
	%
	%   Tracking Snapshot Sequence 4
	%
	% /////////////////////////////////////////////////////////////////////// %

	snapShot = 0;

	if (snapShot)
	    
	    figure; set(gca, 'Box', 'On');
	    iptsetpref('ImshowBorder', 'tight')
	    
	    % Length of Trail History (in frames)
	    trailLength = (1 * 15);
	    
	    % Snapshot frame number
	    for k = ceil(29/T) + [1 2 3 4 5 6]*5
		
		k
		
		% Display the video frame
		% imshow(imcomplement(rgb2gray(read(video, k))));
		%imshow(rgb2gray(read(video, k)));           %MODIFICADO(matlab->octave)
		imshow(rgb2gray(readFrame(video, k)));       %MODIFICADO(matlab->octave)
		% imshow(rgb2gray(read(video, k)));
		hold on;
		set(gcf, 'Position', [255 90 955 715]);
		
		numTentativeTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 1),1)));
		
		numConfirmedTracks(k) = length(unique(TrackRecord(...
		    find(TrackRecord(:,4) == k & TrackRecord(:,2) == 2),1)));
		
		
		% Find the Index to the data at time k for confirmed tracks
		timeIdx = find(TrackRecord(:,4) == k & ...
		    TrackRecord(:,2) == 2);
		
		% Find the tracks at time k
		trackIdx = unique(TrackRecord(timeIdx,1));
		
		
		% Plot each track up to time k
		for trk = trackIdx'
		    
		    % Get the indices to the data for this track up to time k
		    dataIdx = find(TrackRecord(:,1) == trk & ...
		        TrackRecord(:,4) <= k);
		    
		    % Predicted State, Covariance and Residual Covariance
		    SpMat(1,1) = TrackRecord(dataIdx(end), 22);
		    SpMat(1,2) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,1) = TrackRecord(dataIdx(end), 23);
		    SpMat(2,2) = TrackRecord(dataIdx(end), 24);
		    
		    
		    % Plot the last trailLength# track points up to time k
		    if (length(dataIdx) <= trailLength)
		        posX = TrackRecord(dataIdx,5) .* um2px;
		        posY = TrackRecord(dataIdx,6) .* um2px;
		        measX = TrackRecord(dataIdx,19) .* um2px;
		        measY = TrackRecord(dataIdx,20) .* um2px;
		    else
		        posX = TrackRecord(dataIdx(end-trailLength:1:end),5) .* um2px;
		        posY = TrackRecord(dataIdx(end-trailLength:1:end),6) .* um2px;
		        measX = TrackRecord(dataIdx(end-trailLength:1:end),19) .* um2px;
		        measY = TrackRecord(dataIdx(end-trailLength:1:end),20) .* um2px;
		    end
		    
		    % Plot the Tracks and Measurements up to time k
		    plot(measX, measY, 'g')
		    plot(measX, measY, 'g.', 'MarkerSize', 5);
		    plot(posX , posY, 'b');
		    plot(measX(end), measY(end), 'r+');
		    
		    % Plot the track gate
		    % ellipHand = plotEllipse([posX(end); posY(end)], gx * SpMat .* um2px^2);
		    % set(ellipHand, 'Color', 'c');
		    
		    % Label the Track Number
		    text(posX(end)+5, posY(end)-5, num2str(trk), ...
		        'FontName', 'Arial', 'FontSize', 12, 'FontWeight', 'Bold', 'Color', 'k');
		    
		end
		
		% Zoom to frame
		set(gcf, 'Position', [48   607   201   198])
		% axis([330 330+150 25 25 + 150])
		% axis([360 360+150 120 120+150])
		% axis([180 180+150 108 108+150]);%
		% axis([25 25+150 275 275+150]);
		% axis([380 380+150 260 260+150]);
		% axis([400 400+150 100 100+150]);
		axis([280 280+150 220 220+150]);
		
		set(gcf, 'PaperPositionMode', 'Auto')
		set(gcf, 'renderer', 'painters');
		print(gcf, '-dpng', '-r300', [videoFile, 'MTT', num2str(mttAlgorithm), '_Seq4SnapshotFrame', num2str(k), '.png'])
		
	    end
	    
	end

end


