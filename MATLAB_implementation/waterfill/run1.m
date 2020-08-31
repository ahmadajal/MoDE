close all;clear;clc;
%Pairwise distances among two consecutive series
    %A: matrix of time series
load('networkData_12000x64_fft.mat');%Matrix b: 12000x64
load('queries_400x64_fft.mat');%Matrix queries: 400x64
[nd, N] = size(b);
[nq, N] = size(queries);    
n_d = 2000;% number of random data to keep    
n_q = 20;% number of random queries to keep    
pd = randperm(nd);
pq = randperm(nq);
X_array = b(pd(1:n_d),:);
Q_array = queries(pq(1:n_q),:);
    %Number of coefficients maintained
numCoeffs = [8, 16];
%Three options tested: First_Error, Best_Error, Double_waterfilling
numAlgos=3;

%Upper / lower bounds and true distance
    %Dimensions: 1: query, 2:datapoint, 3:algo, 4:numCoeffs
lb = zeros(n_q,n_d,length(numCoeffs),numAlgos);
ub = zeros(n_q,n_d,length(numCoeffs),numAlgos);
dist = zeros(n_q,n_d);
iter = 0;
Iter = n_q*n_d*length(numCoeffs);
tic;
for i = 1:n_q
    Q = Q_array(i,:)';
    for j=1:n_d
        X = X_array(j,:)';
        dist(i,j) = dist_true(X,Q);
        for k=1:length(numCoeffs)
            M = numCoeffs(k);
            %M_f = min(ceil(M*1.15),M);
            M_f = M;
            [lb_f,ub_f] = dist_cc(X,Q,M_f,'First_Error'); 
            [lb_b,ub_b] =  dist_cc(X,Q,M,'Best_Error');
            [lb_dw,ub_dw] = dist_cc(X,Q,M,'Optimal','Double_waterfilling');
            lb(i,j,k,1) = lb_f;
            lb(i,j,k,2) = lb_b;
            lb(i,j,k,3) = lb_dw;
            ub(i,j,k,1) = ub_f;
            ub(i,j,k,2) = ub_b;
            ub(i,j,k,3) = ub_dw;
            iter = iter+1;
            fprintf('%d / %d iterations completed\n', iter , Iter);
        end
    end
end
lb=real(lb);
ub=real(ub);
toc;
save('queries.mat','lb','ub');