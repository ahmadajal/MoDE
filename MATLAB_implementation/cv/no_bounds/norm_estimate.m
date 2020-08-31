function norm_est = norm_estimate(X,numCoeffs,option)
% option: 'First' or 'Best'
% norm from compressed info (coefficients kept)
    % see dist_cc for additional documentation
%------------------------------------------%

% N = length(X);% = length(Q)

% For avoiding to store complex conjugates
    % see dist_cc
X(end) = X(end)/sqrt(2);
power_X = abs(X);%magnitudes of coefficients

%bestCoeffs: coefficients stored, otherCoeffs: coefficients discarded
if strcmp(option,'First') 
    bestCoeffs_X  = 1:numCoeffs;
    %otherCoeffs_X = numCoeffs+1:N;
elseif strcmp(option,'Best') 
    [~,ind_X] = sort(power_X, 'descend');
    bestCoeffs_X  = ind_X(1:numCoeffs);
    %otherCoeffs_X = ind_X(numCoeffs+1:N);
end

norm_est = sqrt(2)*norm(X(bestCoeffs_X));