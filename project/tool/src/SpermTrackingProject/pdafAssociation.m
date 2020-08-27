function [beta] = pdafAssociation(LR, PD, PG)

% Size of the problem matrix
[m, n] = size(LR);

% Initialize beta matrix to zeros
beta = zeros(m,n);

% Likelihood Ratio
% LR = (lam_f^-1) .* f .* PD;

% Calculate beta_jt
for t = 1:n
    
    for j = 1:m               
                               
        beta(j,t) = LR(j,t) / (1 - PD * PG + sum(LR(:,t)) );

    end
    
end

