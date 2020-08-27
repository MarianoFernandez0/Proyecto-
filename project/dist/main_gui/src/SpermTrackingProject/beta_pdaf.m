function [beta] = beta_pdaf(L, PD, PG)
%
%   PDAF beta calculator
%   
%   Usage:  
%   
%       beta(j,t) = beta_pdaf(L, PD, PG)
%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Association Probability
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Size of the problem matrix
[m, n] = size(L);

% Initialize beta matrix to zeros
beta = zeros(m,n);

% Calculate the probability of associating m measurements to n targets
for t = 1:n
    
    for j = 1:m
                        
        beta(j,t) = L(j,t) / (1 - PD * PG + sum(L(:,t)) );        

    end
    
end

