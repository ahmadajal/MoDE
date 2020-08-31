function [X_2d]= FIDE_Gc(X,Score,k_FIDE,Preprocess,MaxIter,Precision)
% This function performs FIDE on X to generate a 2D embedding using Gc.

%Inputs:
    % X: Original data in higher dimension (mxn)
    % Score: the corresponding score vector (mx1)
    % k_FIDE: k nearest neighbours of FIDE, this is also used in Isomap, LLE and UMAP as their parameter k.
    % Preprocess: 'Yes' - preprocess data using min-max normalization.
    % MaxIter: maximum iterations number for FIDE
    % Precison: Precision for Termination Criteria in DLS
%Output:
    % X_2d: 2d Embedding

if strcmp(Preprocess,'Yes')
    X = (X - min(X)) ./ (max(X) - min(X));
elseif strcmp(Preprocess,'No')
    X = X;
else
    error('Error: Preprocess option should be Yes or No')
end


CDM = squareform(pdist(X,'cosine'));
[A,~] = adjmat(CDM,k_FIDE); % create adjacency matrix k-NNG
IncMat = incmat(A); % create incidency matrix
Temp = arrayfun(@(n) 1 - CDM(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));
R = acos(Temp);% Calculate absolute angle measurement
Sign_2ndC = arrayfun(@gt,IncMat*Score,zeros(size(IncMat,1),1)); % This is the sign of angle value
s = 2*Sign_2ndC-1;  % so it's -1 and 1 instead of 0/1
y_angle = s .* R';

X_2d_TEMP = DLS(y_angle,IncMat,Score,MaxIter,Precision); %Apply DLS

norms = sqrt(sum(X.^2,2)); % norm-preserving

X_2d = [norms.*cos(X_2d_TEMP), norms.*sin(X_2d_TEMP)]; 
    






