function [X_2d, error_progression, DM, info]= CV_no_bounds(X,Score,k,MaxIter,Precision,NumCoeffs)
% see CV2 for documentation
    
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
        fx_short = fx(1:((dim+1)/2));
        fx_short = [fx_short(2:end), fx_short(1)]; % bring DC at the end -- as required by dist_cc 
    end
    
    X_f(i,:) = fx_short;
    
end

% Correlation/distance from compressed data
CM = eye(N,N);
DM = zeros(N,N);

for i = 1:N
    for j = i+1:N
        [corr,dist] = corr_dist_no_bounds(X_f(i,:)', X_f(j,:)', NumCoeffs, 'Best');
        %[corr,dist] = corr_dist_no_bounds(X_f(i,:)', X_f(j,:)', NumCoeffs, 'First');
        CM(i,j) = corr;
        DM(i,j) = dist;
    end
end

% make symmetric 
CM = make_symmetric(CM);
DM = make_symmetric(DM);

% Create data graph (kNNG wrt distance)
[A,~] = adjmat(DM,k); % create adjacency matrix k-NNG
inc_mat = incmat(A); % create incidence matrix
Sign_2ndC = arrayfun(@gt,inc_mat*Score,zeros(size(inc_mat,1),1)); % This is the sign of angle value
                                                                % <=> sign in generating incidence matrix        
s = 2*Sign_2ndC-1;  % -> convert to -1/1 from 0/1

% Bounds on correlation (vectors of length = # edges)
C_LB = arrayfun(@(n) CM(inc_mat(n,:)==-1,inc_mat(n,:)==1),1:size(inc_mat,1));
C_UB = C_LB;

R_LB = acos(C_UB);
R_UB = acos(C_LB); 

%y_angle = sort([s .* R_LB', s .* R_UB'],2); % sort mx2 matrix (each row in increasing order)
y_angle = [s .* R_LB', s .* R_UB']; % no need to sort as they are equal

%% =>
% y_angle(:,1) : LB
% y_angle(:,2) : UB
    %% equal
%%

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
    %% norms not known in this scheme
%norms = sqrt(sum(X.^2,2));  % norm-preserving
norms=zeros(N,1);
for i=1:N
    norms(i) = norm_estimate(X_f(i,:)', NumCoeffs, 'Best');% norm of coefficients kept
    %norms(i) = norm_estimate(X_f(i,:)', NumCoeffs, 'First');% norm of coefficients kept
end
X_2d = [norms.*cos(X_2d_TEMP), norms.*sin(X_2d_TEMP)];

% Save information to avoid re-running pre-processing
%
info.DM_LB = DM;
info.DM_UB = DM;
info.CM_LB = CM;
info.CM_UB = CM;
info.A = A;
info.inc_mat = inc_mat;
info.y_angle = y_angle;
