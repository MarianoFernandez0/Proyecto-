function [beta] = pdaf_beta(f, A, PD)
%   Probabilistic Data Association Filter (PDAF)
%   Association Probability Calculator
%   
%   Usage:  
%   
%       beta(j,t) = pdaf_beta(f, A, PD)
%
%   

% Number of measurements Nm, number of tracks Nt
[Nm, Nt] = size(A);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Association Probability
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% For each target, calculate the probability of association

for t = 1:Nt
    
    for j = 1:Nm
                
        
        beta(j,t) = f(j,t) / (1 - PD + sum(f(:,t)));
        

    end
    
end
