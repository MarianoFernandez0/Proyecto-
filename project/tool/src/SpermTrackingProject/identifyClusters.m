function [output] = identifyClusters(A)
%
%   Identifies Track Clusters
%
%   By Leonardo F. Urbano
%
%   23 September 2014
%

% clear all
% close all

% THREE CLUSTERS
% A = [0 1 1 0 1 0 0;
%      1 0 0 0 0 1 0;
%      0 0 0 1 0 0 1;
%      0 0 0 0 0 1 0;
%      0 0 0 1 0 0 0;
%      0 0 1 0 1 0 0];

% ONE CLUSTER
% A = [0 1 0 1 1 0 1;
%      1 0 1 0 0 1 0;
%      0 0 0 1 0 0 0;
%      0 1 0 0 1 0 1;
%      1 1 0 0 0 1 0];



% Possible Outcomes
% if A is empty : there are no measurements at all 
%   every track gets its own cluster



[mA, nA] = size(A);
[i,j] = find(A);
if (mA == 1)
    C = [zeros(length(i),1) i' j'];
else
    C = [zeros(length(i),1) i j];
end
C(1,1) = 1;


% // Initialize
output = [];
clusterNumber = 1;
listDiff = 1;
while (sum(C(:,2)) > 0) && (sum(C(:,3)) > 0)
    
    while (listDiff > 0)
        
        list1 = find(C(:,1))';
        
        for m = list1
            
            i = C(m,2);
            foundRows = find(C(:,2) == i);
            C(foundRows,1) = 1;
            
            list2 = find(C(:,1))';
            
            for n = list2
                
                j = C(n,3);
                foundCols = find(C(:,3) == j);
                C(foundCols,1) = 1;
                
            end
            
        end
        
        list3 = find(C(:,1))';        
        listDiff = length(list3) - length(list1);
        
    end
    
    % Write Output
    D = C(list3,:);
    D(:, 1) = clusterNumber;
    output = [output; D];
    clusterNumber = clusterNumber + 1;
    
    % Remove these Indices from the Problem
    C(list3,:) = 0;
    
    % Initialize the Next Problem
    C(find(C(:,2)>0,1,'first'),1) = 1; %cambié "First" por "first"
    
    % Reset the While Flag
    listDiff = 1;
    
end


% If there are no clusters found then give each track its own cluster
if isempty(output)
    
    for ttt = 1:nA
        output = [output; ttt 0 ttt];
    end
        
else
    % If at least one cluster is found, then give the remaining tracks
    % their own clusters
    tNoMeas = setdiff(1:nA, output(:,3));
    if ~isempty(tNoMeas)
        
        for ttt = tNoMeas
            output = [output; [max(output(:,1))+1 0 ttt]];
        end
        
    end    
end





% output


