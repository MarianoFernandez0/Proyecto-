function [SOLUTIONS, VALUES] = murtys_best(P0, S0, V0, N)
% 1. Find the best solution S0 to P0.
% [S0, V0] = munkres(P0);

% 2. Initialize the list of problem / solution pairs with P0, S0, V0
P{1} = P0; S{1} = S0; V(1) = V0; 

% 3. Clear the list of solutions to be returned
SOLUTIONS = [];
VALUES = [];

% 4. For k = 1 to N (or until the list of P/S pairs is empty)
for k = 1 : N
    
    % 4.1 Find the solution with the best (minimum) value V
    best_idx = find(V == min(V), 1);
    
    % If its not empty
    if ~isempty(best_idx)
        P_BEST = P{best_idx};
        S_BEST = S{best_idx};
        V_BEST = V(best_idx);
        
        % 4.2 Remove <P,S> from the list of P/S pairs
        P{best_idx} = [];
        P(cellfun(@(P) isempty(P), P)) = [];
        S{best_idx} = [];
        S(cellfun(@(S) isempty(S), S)) = [];
        V(best_idx) = [];
        
        % 4.3 Add S to the list of solutions to be returned
        SOLUTIONS{end+1} = S_BEST;
        VALUES{end+1} = V_BEST;
        
        % 4.4 For each assignment in S
        [i, j] = find(S_BEST);
        for n = 1:length(i)
            
            % 4.4.1 Let P' = P
            P_PRIME = P_BEST;
            
            % 4.4.2 Remove this n-th assignment from P_PRIME
            P_PRIME(i(n), j(n)) = Inf;
            
            % 4.4.3 Re-run Munkres on P' to obtain S' and V'
            [S_PRIME, V_PRIME] = munkres(P_PRIME);
            
            % 4.4.4 If S' exists
            if (sum(S_PRIME(:)) == length(i))
                
                % 4.4.4.1 Add <P',S'> to the set of P/S pairs
                P{end+1} = P_PRIME;
                S{end+1} = S_PRIME;
                V(end+1) = V_PRIME;
                
            end
            
            % 4.4.5 From P_BEST, clear the rows and columns from the n-th
            % assignment but leave the assignment intact
            a_ij = P_BEST(i(n), j(n));
            P_BEST(i(n), :) = Inf;
            P_BEST(:, j(n)) = Inf;
            P_BEST(i(n), j(n)) = a_ij;
            
        end % // end loop
        
    else                
        break   % No more solutions - exit the subroutine        
    end    
end

