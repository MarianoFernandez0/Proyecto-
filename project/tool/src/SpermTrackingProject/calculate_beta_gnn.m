function [beta] = calculate_beta_gnn(A)
%   Calculate beta for GNN algorithn
%   
%   Usage:  
%   
%       beta(j,t) = calculate_beta_gnn(A)
%   


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Determine Initial Optimal Assignment / Joint Event
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% Number of measurements Nm, number of tracks Nt
[Nm, Nt] = size(A);

% Initial optimal assignment
THETA = munkres(A);


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%
%   Calculate the Joint Association Probability
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

beta = THETA;

% For each target, calculate the probability of association

% beta = zeros(Nm, Nt);
% 
% for t = 1:Nt
%     
%     for j = 1:Nm
%                 
%         beta(j,t) = beta(j,t) + THETA(j,t);
%                 
%     end
%     
% end
