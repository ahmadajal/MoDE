function [corr,dist] = corr_dist_no_bounds(X,Q,numCoeffs,option)
% option: 'First' or 'Best'
% corr, dist: correlation and distance from compressed info
    % see dist_cc for additional documentation
%------------------------------------------%

% N = length(X);% = length(Q)

% For avoiding to store complex conjugates
    % see dist_cc
X(end) = X(end)/sqrt(2);
Q(end) = Q(end)/sqrt(2);
power_X = abs(X);%magnitudes of coefficients
power_Q = abs(Q);%magnitudes of coefficients

%bestCoeffs: coefficients stored, otherCoeffs: coefficients discarded
if strcmp(option,'First') 
    bestCoeffs_X  = 1:numCoeffs;
    %otherCoeffs_X = numCoeffs+1:N;
    bestCoeffs_Q  = 1:numCoeffs;%same as for X
    %otherCoeffs_Q = numCoeffs+1:N;
elseif strcmp(option,'Best') 
    [~,ind_X] = sort(power_X, 'descend');
    bestCoeffs_X  = ind_X(1:numCoeffs);
    %otherCoeffs_X = ind_X(numCoeffs+1:N);
    [~,ind_Q] = sort(power_Q, 'descend');
    bestCoeffs_Q  = ind_Q(1:numCoeffs);
    %otherCoeffs_Q = ind_Q(numCoeffs+1:N);
end

P0 = intersect(bestCoeffs_X,bestCoeffs_Q);%coefficients where X_i,Q_i known
%P1 = intersect(otherCoeffs_X,bestCoeffs_Q);%coefficients where Q_i known, X_i unknown
%P2 = intersect(bestCoeffs_X,otherCoeffs_Q);%coefficients where X_i known, Q_i unknown
%%P3 = intersect(otherCoeffs_X,otherCoeffs_Q);%coefficients where X_i,Q_i unknown


    % sqrt(2) cancel out due to division
corr = real(X(P0)'*Q(P0)) / ( norm(X(bestCoeffs_X)) * norm(Q(bestCoeffs_Q))); 
% % % Another option
% % corr = real(X(P0)'*Q(P0)) / ( norm(X(P0)) * norm(Q(PO))); 

dist = sqrt(2)* sqrt( norm(X(bestCoeffs_X))^2 + norm(Q(bestCoeffs_Q))^2 - 2*real(X(P0)'*Q(P0)) );
