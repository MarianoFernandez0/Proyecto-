function [beta] = nnAssociation(LR)

% Size of the problem matrix
[m, n] = size(LR);

% Initialize beta matrix to zeros
beta = zeros(m,n);

% Negative Log-likelihood Ratio
NLLR = -log(LR);

% Calculate beta_jt
for t = 1:n
    
    % Nearest Neighbor (measurement with maximum likelihood)
    j = find(NLLR(:,t) == min(NLLR(:,t)));
    
    % Assocation Matrix
    if length(j) > 1
        beta(j,t) = 0;
    else        
        beta(j,t) = 1;
    end
    
    
end

