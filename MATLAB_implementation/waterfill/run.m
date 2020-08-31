close all;clear;clc;
%Pairwise distances among two consecutive series
    %A: matrix of time series
% A = load('data/weblog_100x1024.txt');

dataframe = load('data/stockdata128.mat');

A = dataframe.StockData128; % 436x128 vector
A = make_unit_norm(A);


%A = A(1:10,:);
%A = abs(randn(6,1024));
[n, N] = size(A);
%p = randperm(n);
%A = A(p,:);

    %Number of coefficients maintained
%numCoeffs = 4:4:20; 
%numCoeffs = 200;
%numCoeffs = 8;
%numCoeffs = ceil(N/2*0.8); 
%numCoeffs = [4, 10, 50, 200, 500];
numCoeffs = [2, 4, 8, 16, 32];

%Five options tested; First_Error, Best_Error, Double_Waterfilling_old, Numerical, Double_waterfilling

    %Execution time for the various algorithms
time = zeros(4,length(numCoeffs));
    %Average lower/upper bounds for the various algorithms
lb = zeros(4,length(numCoeffs));
ub = zeros(4,length(numCoeffs));
    %True distance matrix
dist = zeros(n-1,1);

%Scan all time series
for i = 1:n-1
    x = A(i,:);
    y = A(i+1,:);
    
    %Normalization
    x = (x - mean(x))/norm(x);
    y = (y - mean(y))/norm(y);
    dist(i,1) = norm(x-y);
    fx = getNormalFFT(x);
    fy = getNormalFFT(y);
    
    %Remove DC, keep first half of them including the middle one
    X = fx(2:(length(x)/2 +1))';    Y = fy(2:(length(y)/2 +1))';
    
    %Number of coefficients to maintain
    for k = 1:length(numCoeffs)
        M = numCoeffs(k);
        M_f = min(ceil(M*1.15),M);
        tic; [lb_f,ub_f] = dist_cc(X,Y,M_f,'First_Error'); tim_f = toc;
        tic; [lb_b,ub_b] =  dist_cc(X,Y,M,'Best_Error'); tim_b = toc;  
        %tic; [lb_dw_old,ub_dw_old] =  dist_cc(X,Y,M,'Optimal','Double_waterfilling_old'); tim_dw_old = toc;  
        %tic; [lb_cvx,ub_cvx] = dist_cc(X,Y,M,'Optimal','Numerical'); tim_cvx=toc;
        
        tic; [lb_dw,ub_dw] = dist_cc(X,Y,M,'Optimal','Double_waterfilling');tim_dw=toc;
        %tic; [lb_f,ub_f] = dist_uc(X,Y,M_f,'First_Error'); tim_f = toc;
        %tic; [lb_b,ub_b] = dist_uc(X,Y,M,'Best_Error'); tim_b = toc;
        %tic; [lb_cvx,ub_cvx] = dist_uc(X,Y,M,'Optimal_cvx'); tim_cvx=toc;
        %tic; [lb_dw,ub_dw] = dist_uc(X,Y,M,'Optimal'); tim_dw=toc;
            %Update time matrix
        time(1,k) = time(1,k) + tim_f;
        time(2,k) = time(2,k) + tim_b;
        %time(3,k) = time(3,k) + tim_dw_old;
        time(3,k) = time(3,k) + 1000;
        time(4,k) = time(4,k) + tim_dw;
            %Update lower bound matrix
        lb(1,k) = lb(1,k) + lb_f;
        lb(2,k) = lb(2,k) + lb_b;
        %lb(3,k) = lb(3,k) + lb_dw_old;
        lb(3,k) = lb(3,k) + 1000;
        lb(4,k) = lb(4,k) + lb_dw;
            %Update upper bound matrix
        ub(1,k) = ub(1,k) + ub_f;
        ub(2,k) = ub(2,k) + ub_b;
        %ub(3,k) = ub(3,k) + ub_dw_old;
        ub(3,k) = ub(3,k) + 1000;
        ub(4,k) = ub(4,k) + ub_dw;
        fprintf('%d / %d iterations completed\n',(i-1)*length(numCoeffs)+k , (n-1)*length(numCoeffs));
    end
end

time = time / (n-1);
dist = mean(dist);
lb = real(lb) / (n-1);
ub = real(ub) / (n-1);

impr = (ub - lb) ./ (ones(4,1)*(ub(4,:)-lb(4,:)));

%Rounding accuracy
accur = 10^-3;
time = accur*round(time / accur);
dist = accur*round(dist / accur);
lb = accur*round(lb / accur);
ub = accur*round(ub / accur);
impr = accur*round(impr / accur)

