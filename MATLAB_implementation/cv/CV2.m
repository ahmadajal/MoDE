function [X_2d, error_progression, DM, info]= CV2(X,Score,k,MaxIter,Precision,NumCoeffs, use_same_LB_UB, compression)
% This function performs CV on X to generate a 2D MoDE embedding using Gd.

%Inputs:
    % X:                Original data in higher dimension (mxD)
    % Score:            the corresponding score vector (mx1)
    % k:                # nearest neighbors for data graph in MoDE
    % MaxIter:          maximum iterations number for DLS_gradient
    % Precison:         Precision for Termination Criteria in DLS_gradient
    % NumCoeffs:        number of coefficients to use for compression
    % use_same_LB_UB:   if 1, use average distance bounds to get the same LB/UB
%                       %% => ls problem
%Output:
    % X_2d:                 2d Embedding
    % error_progression:    error of DLS_gradient, per iteration
    % DM:                   average distance matrix (LB+UB)/2
    % info:                 additional information as output
    % -- useful to re-run algos without preprocessing
    % Fields:
    % DM_LB, DM_UB -- (lower/upper bound distance matrices)
    % CM_LB, CM_UB -- (lower/upper bound correlation matrices)
    % A -- (adjacency matrix of average graph)
    % inc_mat -- (incidence matrix of average graph)
    % y_angle -- (lower/upper angular values in edges of average graph)
    % with sign resolution (to be used by DLS_gradient)
    % y_angle(:,1) is lb
    % y_angle(:,2) is ub

N = size(X,1); % size of dataset
dim = size(X,2); % length of time series
fprintf('number of data points: %d \n', N);
% X_f holds the Fourier representation of X
    %%%%% Does not store complex conjugates -- for maximal compression %%%%%%
        % Uses the symmetry of FFT
%
    %%% Checks two cases (odd/even) as required by dist_cc (see lines 4-9 in dist_cc.m)
        %%
        % for even it stores n/2+1 (includes DC-- can be removed in runme via subtracting the mean)
        % for odd it stores (n+1)/2 -- DC given as last entry
    %%%
if compression
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
                           'Optimal','Double_waterfilling');
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
  end
else
  DM_UB = L2_distance2(X', X',1); %exact distance matrix
  DM_LB = DM_UB;
  CM_LB = eye(N,N); CM_UB = eye(N,N);
  % save the norms to use later
  row_norms = zeros(N);
  for i = 1:N
      row_norms(i) = norm(X(i, :));
  end
  for i = 1:N
      for j = i+1:N
          norm_xi = row_norms(i);
          norm_xj = row_norms(j);

          if norm_xi*norm_xj == 0
              norm_xi;
              norm_xj;
              error('Remove zero points from dataset');
          end
          dist_avg = DM_UB(i,j); % since DM_UB and DM_LB are the same (exact distances)
          CM_LB(i,j) = (norm_xi^2 + norm_xj^2 - dist_avg^2) / (2*norm_xi*norm_xj);
          CM_UB(i,j) = CM_LB(i,j);
      end
  end
end

% make symmetric
if compression
  DM_LB = make_symmetric(DM_LB);
  DM_UB = make_symmetric(DM_UB);
end

CM_LB = make_symmetric(CM_LB);
CM_UB = make_symmetric(CM_UB);

% Create data graph (kNNG)
DM = (DM_LB + DM_UB)/2; % take mid-point distance
[A,~] = adjmat(DM,k); % create adjacency matrix k-NNG
inc_mat = incmat(A); % create incidence matrix
Sign_2ndC = arrayfun(@gt,inc_mat*Score,zeros(size(inc_mat,1),1)); % This is the sign of angle value
                                                                % <=> sign in generating incidence matrix
s = 2*Sign_2ndC-1;  % -> convert to -1/1 from 0/1

% Bounds on correlation (vectors of length = # edges)
C_LB = arrayfun(@(n) CM_LB(inc_mat(n,:)==-1,inc_mat(n,:)==1),1:size(inc_mat,1));
C_UB = arrayfun(@(n) CM_UB(inc_mat(n,:)==-1,inc_mat(n,:)==1),1:size(inc_mat,1));

% Bounds on angular difference
    % Note: acos() is decreasing
            % C_UB -> R_LB
            % C_LB -> R_UB
%CHANGE: I added real to the following 2 lines
R_LB = real(acos(C_UB));
R_UB = real(acos(C_LB));

%%%%% Alternative averaging below for CV (becomes ls problem)
%if (use_same_LB_UB == 1)
%    R_LB = 1/2*(R_LB + R_UB);
%    R_UB = R_LB;
%end

% Sort to make sure LB<UB
    % This handles the following, altogether:
        % 1. -dist^2 is decreasing
        % 2. acos is decreasing
        % 3. multiplies with sign

y_angle = sort([s .* R_LB', s .* R_UB'],2); % sort mx2 matrix (each row in increasing order)
%% =>
% y_angle(:,1) : LB
% y_angle(:,2) : UB
%%

%%Alternative formulation
%y_angle = [R_LB',R_UB'];
%inc_mat = diag(s)*inc_mat;
    %% This is equivalent given that |x - (x)_l^u| = |-x - (-x)_{-u}^{-l}|
                                     %% -- i.e., using sign to order the two vertices in an edge
    %%%%% For some  reason this makes DLS_gradient (much) slower - NOT USED %%%%%

% Run gradient method
[X_2d_TEMP, error_progression] = DLS_gradient(y_angle(:,1), y_angle(:, 2), ...
                                                inc_mat, ...
                                                Score, ...
                                                MaxIter, ...
                                                Precision, ...
                                                k); % Gradient method


%%%%%%%%% OTHER METHODS - NOT USED

%[X_2d_TEMP, error_progression] = DLS2(y_angle(:,1), y_angle(:, 2),inc_mat,Score,MaxIter,Precision);
                                  % Random projections

%[X_2d_TEMP, error_progression] = DLS_new(y_angle(:,1), y_angle(:, 2),inc_mat,Score,MaxIter,Precision);
                                  % Stochastic gradient with averaging

% % %%% SAG(A) %%%
% % %---------------------------------------------------%
% % % Same number of passes as the gradient method (m x)
% % m = size(inc_mat,1);
% % MaxIter = MaxIter*m;
% %
% % %Algo = 'SAGA';
% % Algo = 'SAG';
% % %precon = 'Y';
% % precon = 'N';
% %
% % [X_2d_TEMP, error_progression] = DLS_SAG_A(y_angle(:,1), y_angle(:, 2),inc_mat,Score,MaxIter,Precision,Algo,precon,'edge');
% %                                  % SAG(A) method
% % %---------------------------------------------------%

%%%%%%%%% END OF OTHER METHODS - NOT USED


% Visualization using norms
norms = sqrt(sum(X.^2,2));  % norm-preserving
X_2d = [norms.*cos(X_2d_TEMP), norms.*sin(X_2d_TEMP)];

% Save information to avoid re-running pre-processing
%
info.DM_LB = DM_LB;
info.DM_UB = DM_UB;
info.CM_LB = CM_LB;
info.CM_UB = CM_UB;
info.A = A;
info.inc_mat = inc_mat;
info.y_angle = y_angle;
