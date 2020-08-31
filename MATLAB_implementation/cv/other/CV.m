function [X_2d, error_progression, DM]= CV(X,Score,k_FIDE,MaxIter,Precision,NumCoeffs, k, use_same_LB_UB)
% This function performs CV on X to generate a 2D embedding using Gd.

%Inputs:
    % X:            Original data in higher dimension (mxD)
    % Score:        the corresponding score vector (mx1)
    % k_FIDE:       k nearest neighbours of CV, this is also used in Isomap, LLE and UMAP as their parameter k.
    % MaxIter:      maximum iterations number for DLS_gradient
    % Precison:     Precision for Termination Criteria in DLS_gradient
    % NumCoeffs:    number of coefficients to use for compression 
%    
%Output:
    % X_2d:                 2d Embedding
    % error_progression:    error of DLS_gradient
    % DM:                   distance matrix (LB+UB)/2
    
    
% Pre-processing
norms = sqrt(sum(X.^2,2));  % norm-preserving --  store initial norms
X = make_unit_norm(X);      % can normalize as CV depends on correlation bounds that do not depend on norms
                                %% greatly simplifies the code below

% X_f holds the Fourier representation of X
    %%%%% Does not store complex conjugates
          % Checks two cases (odd/even) as required by dist_cc

dim = size(X,2); % length of time series   

%%
% for even it stores n/2+1 (includes DC-- can be removed in runme via subtracting the mean)
% for odd it stores (n+1)/2
    % Uses the symmetry of FFT  
X_f = zeros(size(X,1), ceil(dim/2+1));

% Check if length is even
is_even = ~(mod(dim,2));
max_error=0;

for i = 1:size(X,1)
    x = X(i,:);
    fx = getNormalFFT(x); %dimensions 1xdim
    
    % Include DC in all cases
        % Can subtract mean in runme to remove DC as preprocessing
    if is_even    
        fx_short = fx(1:(dim/2 +1));
        fx_short(1) = fx_short(1)/sqrt(2); % divide DC by sqrt(2) -- as required by dist_cc
    else %is odd
        fx_short = fx(1:((dim+1)/2 +1));
        fx_short = [fx_short(2:end), fx_short(1)]; % bring DC at the end -- as required by dist_cc 
    end
        
%     % Including DC
%     %fx_short = fx(1:(length(x)/2 +1)); %dimensions 1x(D/2+1)   
%     
%     % This is without DC
%     %fx_short = fx(2:(length(x)/2 +1)); %dimensions 1x(D/2)   
    
    X_f(i,:) = fx_short;
    
    temp = fx_short;
    temp(end) = temp(end)/sqrt(2);
    max_error = max([max_error, abs(sqrt(2)*norm(temp) - norm(X(i,:)))]);
end
max_error
% LB and UB for L2 distances
N = size(X,1);
DM_LB = zeros(N,N); DM_UB = zeros(N,N);

for i = 1:N
    for j = i+1:N
        [lb,ub] = dist_cc(X_f(i,:)', ...
                          X_f(j,:)', ...
                          NumCoeffs, ...
                         'Optimal','Double_waterfilling');
                         % 'First_Error');
                        %'Best_Error');
        % Ignore imag. parts 
            % due to numerical accuracy can get super
            % small imag. parts as opposed to 0
        DM_LB(i,j) = real(lb); 
        DM_UB(i,j) = real(ub);        
    end
end
% make symmetric 
DM_LB = make_symmetric(DM_LB);
DM_UB = make_symmetric(DM_UB);

DM = (DM_LB + DM_UB)/2; % take mid-point distance
[A,~] = adjmat(DM,k_FIDE); % create adjacency matrix k-NNG
IncMat = incmat(A); % create incidence matrix
Sign_2ndC = arrayfun(@gt,IncMat*Score,zeros(size(IncMat,1),1)); % This is the sign of angle value
s = 2*Sign_2ndC-1;  % so it's -1 and 1 instead of 0/1

% Correlation 
%% dist^2(x_i,x_j) = ||x_i||^2 + ||x_j^2|| - 2*corr_ij*||x_i|*||x_j||
        %% =>
        %% corr_ij =  (||x_i||^2 + ||x_j^2|| - dist^2(x_i,x_j)) / (2*||x_i|*||x_j||)
   %%%% for normalized data this simplifies to 1 - dist^2/2 %%%%

% Bounds on correlation (matrices)   
    % Note:  -dist^2 is decreasing, hence
            % DM_LB -> CM_UB
            % DM_UB -> CM_LB
%            

one = ones(size(DM_LB)) - eye(size(DM_LB));

%CM_UB = 1 - (DM_LB.^2)/2; % when norm=1 per row
%CM_LB = 1 - (DM_UB.^2)/2; % when norm=1

% % % % % % % % % % CM_UB = one - (DM_LB.^2)/2; % when norm=1 per row
% % % % % % % % % % CM_LB = one - (DM_UB.^2)/2; % when norm=1
% % % % % % % % % % 
% % % % % % % % % % % Give same UB/LB -- i.e., FIDE 
% % % % % % % % % % if (use_same_LB_UB == 1)
% % % % % % % % % % %     CM_UB = 1 - DM.^2/2;
% % % % % % % % % % %     CM_LB = 1 - DM.^2/2;    
% % % % % % % % % %     CM_UB = one - DM.^2/2;
% % % % % % % % % %     CM_LB = one - DM.^2/2;    
% % % % % % % % % % end
% % % % % % % % % % 
% % % % % % % % % % 
% % % % % % % % % % % Bounds on correlation (vectors of length m -- # edges)
% % % % % % % % % % C_UB = arrayfun(@(n) CM_UB(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));
% % % % % % % % % % C_LB = arrayfun(@(n) CM_LB(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));

C_UB = arrayfun(@(n) 1 - DM_LB(IncMat(n,:)==-1,IncMat(n,:)==1).^2/2,1:size(IncMat,1));
C_LB = arrayfun(@(n) 1 - DM_UB(IncMat(n,:)==-1,IncMat(n,:)==1).^2/2,1:size(IncMat,1));

mean(C_UB)
mean(C_LB)

% Bounds on angular difference
    % Note: acos() is decreasing
            % C_LB -> R_UB
            % C_UB -> R_LB
R_UB = acos(C_LB); 
R_LB = acos(C_UB);

%mean(R_LB)
%mean(R_UB)

%%%%% Alternative averaging below for CV (becomes FIDE)
%if (use_same_LB_UB == 1)    
%    R_LB = 1/2*(R_LB + R_UB);
%    R_UB = R_LB;
%end

% Sort to make sure LB<UB
    % This handles the following, altogether:
        % 1. -dist^2 is decreasing 
        % 2. acos is decreasing
        % 3. multiplies with sign
%%        
%%        
y_angle = sort([s .* R_LB', s .* R_UB'],2); % sort mx2 matrix (each row in increasing order)
% y_angle(:,1) : LB
% y_angle(:,2) : UB
%%

%%Alternative formulation
%%y_angle = [R_LB',R_UB'];
%IncMat = diag(s)*IncMat;
    %%% For some  reason this makes DLS_gradient (much) slower - NOT USED %%%

% Run gradient method    
[X_2d_TEMP, error_progression] = DLS_gradient(y_angle(:,1), y_angle(:, 2), ...
                                                IncMat, ...
                                                Score, ...
                                                MaxIter, ...
                                                Precision, ...
                                                k); 
                                 % Gradient method
                                 
%%%%%%%%% OTHER METHODS - NOT USED

%[X_2d_TEMP, error_progression] = DLS2(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision); 
                                  % Random projections

%[X_2d_TEMP, error_progression] = DLS_new(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision); 
                                  % Stochastic gradient with averaging
                                 
% % %%% SAG(A) %%%
% % %---------------------------------------------------%                                 
% % % Same number of passes as the gradient method (m x) 
% % m = size(IncMat,1);
% % MaxIter = MaxIter*m;
% % 
% % %Algo = 'SAGA';
% % Algo = 'SAG';
% % %precon = 'Y'; 
% % precon = 'N'; 
% % 
% % [X_2d_TEMP, error_progression] = DLS_SAG_A(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision,Algo,precon,'edge'); 
% %                                  % SAG(A) method
% % %---------------------------------------------------%                                                                

%%%%%%%%% END OF OTHER METHODS - NOT USED


% Visualization using norms
X_2d = [norms.*cos(X_2d_TEMP), norms.*sin(X_2d_TEMP)];




