function [A,A_DM] = adjmat(DM,k)

% Creates the adjacency matrix of a graph from dissimality matrix

%Input:
  %DM:  dissimality matrix (nxn, n: number of vertices)
  %k: retrieve an adjaceny matrix with k nearest neighbor from distance matrix

%Output:
  %A:    adjacency matrix (asymmetric)
  %A_DM: the corresponding A with every 1 entry replaced by its dissimality value
  
if nargin==0
    error('No inputs.')
    return
end

[m,n] = size(DM); %% matrix dimensions

if m~=n
    error('Invalid input: not a square matrix'); % dissimilarity matrix is symmetric
    return
end

sz = size(DM,1);
E = DM + diag(Inf(1,sz)); % Set diagonal element of DM to be Inf, for self-loop remove
[~, mm] = sort(E);


mmi = mm(1:k,:)';  % n smallest distances
dm_idx = sparse(repmat(1:sz,1,k),mmi(:),1,sz,sz); % Create the sparse matrix of adjacency matrix, ith row records the n nearest neighbors position

A_DM = full(DM.*dm_idx); % Element-wise multiplication leaves dissimality value instead of 1
A = full(dm_idx);
