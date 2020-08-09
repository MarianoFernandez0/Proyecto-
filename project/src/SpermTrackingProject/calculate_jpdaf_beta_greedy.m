function [beta] = calculate_jpdaf_beta_greedy(A, PD)
%   Sub-optimal Joint Probabilistic Data Association (JPDA)
%   Joint Association Probability Calculator using greedy assignment to
%   obtain feasible joint events
%   
%   Usage:  
%   
%       beta(j,t) = calculate_jpdaf_beta(f, A, PD)
%
%   


% Convert from negative log likelihoods to likelihoods
% (or does it even matter?)
f = exp(-A);

% Number of measurements Nm, number of tracks Nt
[Nm, Nt] = size(A);

% Original distance matrix
A_orig = A;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Determine Initial Optimal Assignment / Joint Event
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Initial optimal assignment
THETA{1} = munkres(A_orig);

% How many sweeps should we perform? 
num_events = length(find(THETA{1}));


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the N-best joint events using greedy
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% The set of events in the Initial Joint Event
[theta_j, theta_t] = find(THETA{1});

% Create N-1 new events
for N = 1:num_events
    
    % Copy the Original Optimal Assignment
    A_TEMP = A_orig;
    
    % Remove an event
    A_TEMP(theta_j(N), theta_t(N)) = Inf;
    
    % Re-run Munkres
    A_OPT = greedy(A_TEMP);        
    
    % Record the event 
    THETA{end+1} = A_OPT;
    
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Probability of Each Joint Event
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% num_events = 0;

for N = 1:(num_events + 1)
    
    % Hadamard product of A_opt and f
    C_N = THETA{N} .* f;
    
    % First term of P{theta | z^k}
    [theta_j, theta_t, v] = find(C_N);
    term1 = prod(v);
    
    % Second term of P{theta | z^k}
    term2 = (PD)^(length(theta_t)) * (1 - PD)^(Nt - length(theta_t));
    
    % Probability of joint event
    P_THETA{N} = term1 * term2;
    
end

% Calculate the normalization constant c
c = sum(cell2mat(P_THETA));

% Normalize the probabilities
for N = 1:(num_events + 1)
    
    P_THETA{N} = P_THETA{N}/c;
    
end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Joint Association Probability
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% For each target, calculate the probability of association

for t = 1:Nt
    
    for j = 1:Nm
        
        beta(j,t) = 0;

        for N = 1:(num_events + 1)
                    
            beta(j,t) = beta(j,t) + P_THETA{N} * THETA{N}(j,t);
            
        end        

    end
    
end
