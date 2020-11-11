function [beta] = pdafProcess(f, PD, PG, lam_f)

% Size of the problem matrix
[m, n] = size(f);

% Initialize beta matrix to zeros
beta = zeros(m,n);

% Likelihood Ratio
LR = inv(lam_f) * f(j,t) * PD;

% Calculate beta_jt
for t = 1:n
    
    for j = 1:m               
                               
        beta(j,t) = LR(j,t) / (1 - PD * PG + sum(LR(:,t)) );

    end
    
end

