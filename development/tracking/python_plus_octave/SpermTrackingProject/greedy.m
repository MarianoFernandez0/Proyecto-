function [A_opt] = greedy(A)

% clear all 
% close all
% 
% A = randn(5,5)

% Size of A
[m,n] = size(A);

% Assignment matrix
A_opt = zeros(m,n);

% Column index
cols = 1:n;

for j = 1:m

    if (min(A(j,cols)) ~= inf)
    
        [~, t] = find((A == min(A(j,cols))));
    
        A_opt(j,t) = 1;
    
        cols = cols(cols ~= t);
        
    end
           
end
    

