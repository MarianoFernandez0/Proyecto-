function [beta] = calculate_beta_jpda(A)
%   Sub-optimal Joint Probabilistic Data Association (JPDA)
%   Joint Association Probability Calculator using murty's k-best
%   
%   Usage:  
%   
%       beta(j,t) = calculate_jpdaf_beta_murty(f, A, PD)
%
%   

PD = 0.95; 

% Convert from negative log likelihoods to likelihoods
% (or does it even matter?)
f = exp(-A);

% Number of measurements Nm, number of tracks Nt
[Nm, Nt] = size(A);

% Original distance matrix
A_orig = A;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Find N-best assignments using Murty's method
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Initial problem
P0 = A_orig;

% Initial optimal assignment
[S0, V0] = munkres(P0);

SOLUTIONS = murtys_best(P0, S0, V0, 10);

if (length(SOLUTIONS) == 1)

    THETA{1} = S0;

else
    
    THETA = SOLUTIONS;
    THETA{end+1} = S0;

end

num_events = length(THETA);
        

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Probability of Each Joint Event
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

for N = 1:num_events
    
    % Hadamard product of A_opt and f
    C_N = THETA{N} .* f;        
    % C_N = THETA{N} .* A;    
    
    % P{ THETA | z^k}
    [theta_j, theta_t, v] = find(C_N);
    term1 = sum(-log(v));    
    
    % Second term of P{theta | z^k}
    term2 = length(theta_t) * log(PD) + (Nt - length(theta_t)) * log(1-PD);
    %term2 = (PD)^(length(theta_t)) * (1 - PD)^(Nt - length(theta_t));
    
    % Probability of joint event
    P_THETA{N} = term1 + term2;
    
end

% Calculate the normalization constant c
c = sum(cell2mat(P_THETA));

% Normalize the probabilities
for N = 1:num_events
    
    P_THETA{N} = P_THETA{N} - c;
    
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

        for N = 1:num_events
                    
            beta(j,t) = beta(j,t) + P_THETA{N} * THETA{N}(j,t);
            
        end        

    end
    
end
