function [beta] = calculate_pdaf_beta(A, PD)
%   Probabilistic Data Association Filter (PDAF)
%   Association Probability Calculator
%   
%   Usage:  
%   
%       beta(j,t) = calculate_pdaf_beta(f, A, PD)
%
%   

% Number of measurements Nm, number of tracks Nt
[m, n] = size(A);


% Convert from negative log likelihood to likelihood
f = exp(-A);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Association Probability
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% For each target, calculate the probability of association

for j = 1:m
    
    for t = 1:n
                
        
        beta(j,t) = f(j,t) / (1 - PD + sum(f(:,t)));
        

    end
    
end
