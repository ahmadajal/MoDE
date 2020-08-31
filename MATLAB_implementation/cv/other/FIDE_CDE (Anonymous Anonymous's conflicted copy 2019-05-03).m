function [X_2d, error_progression, DM]= FIDE_CDE(X,Score,k_FIDE,Preprocess,MaxIter,Precision,NumCoeffs, k)
% This function performs FIDE on X to generate a 2D embedding using Gd.

%Inputs:
    % X: Original data in higher dimension (mxD)
    % Score: the corresponding score vector (mx1)
    % k_FIDE: k nearest neighbours of FIDE, this is also used in Isomap, LLE and UMAP as their parameter k.
    % Preprocess: 'Yes' - preprocess data using min-max normalization.
    % MaxIter: maximum iterations number for FIDE
    % Precison: Precision for Termination Criteria in DLS
    %%% NumCoeffs: number of coefficients to use for compression (not an input; set below)
%Output:
    % X_2d: 2d Embedding

if strcmp(Preprocess,'Yes')
    X = make_unit_norm(X);
elseif strcmp(Preprocess,'No')
    1;
else
    error('Error: Preprocess option should be Yes or No')
end

% X_f holds the Fourier representation of X
X_f = zeros(size(X,1), size(X,2)/2);
for i = 1:size(X,1)
    x = X(i,:);
    fx = getNormalFFT(x); %dimensions 1xD
    
    % Remove DC, keep first half of them including the middle one
    fx_short = fx(2:(length(x)/2 +1)); %dimensions 1x(D/2)   
    X_f(i,:) = fx_short;
end

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
                        
        DM_LB(i,j) = real(lb); 
        DM_UB(i,j) = real(ub);
    end
end
%Im_LB = sum(sum(abs(imag(DM_LB))))
%Im_UB = sum(sum(abs(imag(DM_UB))))
% make symmetric (& check for imaginary parts).
DM_LB = make_symmetric(DM_LB);
DM_UB = make_symmetric(DM_UB);

DM = (DM_LB + DM_UB)/2; % take mid-point distance
[A,~] = adjmat(DM,k_FIDE); % create adjacency matrix k-NNG
IncMat = incmat(A); % create incidence matrix
Sign_2ndC = arrayfun(@gt,IncMat*Score,zeros(size(IncMat,1),1)); % This is the sign of angle value
s = 2*Sign_2ndC-1;  % so it's -1 and 1 instead of 0/1

% old code
% CDM = squareform(pdist(X,'cosine'));
% Dist_knn = arrayfun(@(n) 1 - CDM(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));
% R = acos(Dist_knn);% Calculate absolute angle measurement

% Give different upper/lower bound
%CDM_LB = (DM_LB.^2)/2; % when norm=1 per row
    % The above line is equal to 1 - corr_ub
%CDM_UB = (DM_UB.^2)/2; % when norm=1
    % The above line is equal to 1 - corr_lb

% Give same UB/LB, ie FIDE. This is correct
CDM_LB = DM.^2/2;
CDM_UB = DM.^2/2;

%%% The below lines give corr_ub, corr_lb respectively
% acos is decreasing function, so it is ok below
innerprod_LB = arrayfun(@(n) 1 - CDM_LB(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));
innerprod_UB = arrayfun(@(n) 1 - CDM_UB(IncMat(n,:)==-1,IncMat(n,:)==1),1:size(IncMat,1));

R_LB = acos(innerprod_LB);% Calculate absolute angle measurement
R_UB = acos(innerprod_UB);% 

%%%%% Alternative averaging below
% R_LB = 1/2*(R_LB + R_UB);
% R_UB = R_LB;

y_angle = sort([s .* R_LB', s .* R_UB'],2);

%%Alternative formulation
%y_angle = [R_LB',R_UB'];
%IncMat = diag(s)*IncMat;

%[X_2d_TEMP, error_progression] = DLS2(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision); 
                                  % Random projections

%[X_2d_TEMP, error_progression] = DLS_new(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision); 
                                  % Stochastic gradient with averaging

[X_2d_TEMP, error_progression] = DLS_gradient(y_angle(:,1), y_angle(:, 2), ...
                                                IncMat, ...
                                                Score, ...
                                                MaxIter, ...
                                                Precision, ...
                                                k); 
                                 % Gradient method
                                 
                                 
% % % %%% SAG(A) %%%
% % % %---------------------------------------------------%                                 
% % % % Same number of passes as the gradient method (m x) 
% % % m = size(IncMat,1);
% % % MaxIter = MaxIter*m;
% % % 
% % % %Algo = 'SAGA';
% % % Algo = 'SAG';
% % % %precon = 'Y'; 
% % % precon = 'N'; 
% % % 
% % % [X_2d_TEMP, error_progression] = DLS_SAG_A(y_angle(:,1), y_angle(:, 2),IncMat,Score,MaxIter,Precision,Algo,precon,'edge'); 
% % %                                  % SAG(A) method
% % % %---------------------------------------------------%                                                                

norms = sqrt(sum(X.^2,2)); % norm-preserving
X_2d = [norms.*cos(X_2d_TEMP), norms.*sin(X_2d_TEMP)];




