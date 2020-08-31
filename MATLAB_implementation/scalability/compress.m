function [DM, DM_LB, DM_UB, CM_LB, CM_UB]= compress(X,NumCoeffs, use_same_LB_UB)
%% This function compresses the data and returns the distance and correlation
%% lower and upper bound matrices
    
N = size(X,1); % size of dataset
dim = size(X,2); % length of time series

% X_f holds the Fourier representation of X
    %%%%% Does not store complex conjugates -- for maximal compression %%%%%%
        % Uses the symmetry of FFT
%
    %%% Checks two cases (odd/even) as required by dist_cc (see lines 4-9 in dist_cc.m)
        %%
        % for even it stores n/2+1 (includes DC-- can be removed in runme via subtracting the mean)
        % for odd it stores (n+1)/2 -- DC given as last entry
    %%%

% Data in Fourier basis (without storing complex conjugates)
X_f = zeros(N, ceil(dim/2+1));

% Check if length is even
is_even = ~(mod(dim,2));

for i = 1:N
    x = X(i,:);
    fx = getNormalFFT(x);

    % Include DC in all cases
        % Can subtract mean in runme to remove DC as preprocessing
    if is_even
        fx_short = fx(1:(dim/2 +1));
        fx_short(1) = fx_short(1)/sqrt(2); % divide DC by sqrt(2) -- as required by dist_cc
    else %if odd
        fx_short = fx(1:ceil(dim/2+1)); %(dim+1)/2), why?
        fx_short = [fx_short(2:end), fx_short(1)]; % bring DC at the end -- as required by dist_cc
    end

%     % Including DC
%     %fx_short = fx(1:(length(x)/2 +1)); %dimensions 1x(D/2+1)
%
%     % This is without DC
%     %fx_short = fx(2:(length(x)/2 +1)); %dimensions 1x(D/2)

    X_f(i,:) = fx_short;

end

% LB and UB for L2 distances & Correlations
DM_LB = zeros(N,N); DM_UB = zeros(N,N);
CM_LB = eye(N,N); CM_UB = eye(N,N);

for i = 1:N
    for j = i+1:N
        [lb,ub] = dist_cc(X_f(i,:)', ...
                          X_f(j,:)', ...
                          NumCoeffs, ...
                         'First_Error','Double_waterfilling'); % previously was optimal
                         %'Best');
                         % 'First_Error');
                         %'Best_Error');
        % Ignore imag. parts
            % due to numerical accuracy can get super
            % small imag. parts as opposed to 0 value
        DM_LB(i,j) = real(lb);
        DM_UB(i,j) = real(ub);

        % Norms of data points
        norm_xi = norm(X(i,:));
        norm_xj = norm(X(j,:));

        if norm_xi*norm_xj == 0
            error('Remove zero points from dataset');
        end

        %% Correlation bounds calculation
        %%
        %% dist^2(x_i,x_j) = ||x_i||^2 + ||x_j^2|| - 2*corr_ij*||x_i|*||x_j||
        %% =>
        %% corr_ij =  (||x_i||^2 + ||x_j^2|| - dist^2(x_i,x_j)) / (2*||x_i|*||x_j||)
            % for normalized data this simplifies to 1 - dist^2/2
        %%

        %%%%% Upper/lower bound on correlation (matrices)
        %
        % Note:  -dist^2 is decreasing, hence
                % DM_UB -> CM_LB
                % DM_LB -> CM_UB
        if (use_same_LB_UB ~= 1)
            CM_LB(i,j) = (norm_xi^2 + norm_xj^2 - DM_UB(i,j)^2) / (2*norm_xi*norm_xj);
            CM_UB(i,j) = (norm_xi^2 + norm_xj^2 - DM_LB(i,j)^2) / (2*norm_xi*norm_xj);
        else % use average distance
            dist_avg = (DM_LB(i,j) + DM_UB(i,j)) / 2;
            CM_LB(i,j) = (norm_xi^2 + norm_xj^2 - dist_avg^2) / (2*norm_xi*norm_xj);
            CM_UB(i,j) = CM_LB(i,j);
        end
    end
    if mod(i, 100) == 0
        fprintf("%d data processed \n", i)
    end
end

% make symmetric
DM_LB = make_symmetric(DM_LB);
DM_UB = make_symmetric(DM_UB);

CM_LB = make_symmetric(CM_LB);
CM_UB = make_symmetric(CM_UB);

DM = (DM_LB + DM_UB)/2; % take mid-point distance
