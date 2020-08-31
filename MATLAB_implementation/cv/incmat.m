function [A] = incmat(B)

% Creates the sparse incidence matrix of a graph from adjacency matrix
% Code adapted from Ondrej Sluciak(2009). Graph adjacency matrix to
% incidence matrix, MATLAB Central File Exchange. 
% (Source: https://www.mathworks.com/matlabcentral/fileexchange/24661-graph-adjacency-matrix-to-incidence-matrix,retrieved Jan 2019)


%Input:
%B: adjacency matrix (nxn, n: number of vertices)

%Output:
%A: sparse incidence matrix 

%% Works for both directed and undirected graphs
    %% self-loops ignored
    %% if both (i,j) and (j,i) are links (e.g., if graph is symmetric)
    %% only one link is taken in the incident matrix: (i,j) with i<j
   
if nargin==0
    error('No inputs.')
    return
end

[m,n] = size(B); %% matrix dimensions

if m~=n
    error('Invalid input: not a square matrix');
    return
end

if ~isempty(find(B~=1 & B~=0,1))
    error('Invalid input: not a 0-1 matrix');
    return
end

B = B - diag(diag(B)); %% ignore self-loops
B = max(B - tril(B'),zeros(n)); %% remove lower diagonal symmetric part of B
% Keep only one of symmetric links (i,j) with i<j
    %%i.e., if B_{ij}=1 & B_{ji}=1, set B_{ij}=1 and B_{ji}=0, for i<j

NodeIndex  = size(B,1);

[x,y] = find(B'); %Find the 1's index from adj matrix     
edgeIndex = length(x);
OnePos = ones(edgeIndex,1); % used as entry value in sparse inc matrix
EdgesNumber = 1:edgeIndex;

% Construct the sparse inc matrix by mapping 1 and -1 to their position
A = sparse([EdgesNumber, EdgesNumber]',... 
           [x; y],... 
           [OnePos; -OnePos],... 
           edgeIndex,NodeIndex); 

       



