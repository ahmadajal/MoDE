function [X_2d, error_progression, DM, info]= modified_CV(X,Score,k,MaxIter,Precision, DM_LB, DM_UB, CM_LB, CM_UB)
% This function performs CV on the upper and lower bound distance and correlation
% matrices to generate a 2D MoDE embedding using Gd.

%Inputs:
    % X:                Original data in higher dimension (mxD)
    % Score:            the corresponding score vector (mx1)
    % k:                # nearest neighbors for data graph in CV
                        %% this is also used in IsoMap, LLE and UMAP as their parameter k.
    % MaxIter:          maximum iterations number for DLS_gradient
    % Precison:         Precision for Termination Criteria in DLS_gradient
    % DM_LB:            lower bound distance matrix
    % DM_UB:            upper bound distance matrix
    % CM_LB:            lower bound distance matrix
    % CM_UB:            upper bound distance matrix

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


% fprintf('number of data points: %d \n', num_data_points);


% make symmetric

% DM_LB = make_symmetric(DM_LB);
% DM_UB = make_symmetric(DM_UB);
%
% CM_LB = make_symmetric(CM_LB);
% CM_UB = make_symmetric(CM_UB);

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
