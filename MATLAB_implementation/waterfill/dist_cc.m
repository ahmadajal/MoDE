function [lb,ub] = dist_cc(X,Q,numCoeffs,option_algo,algo_optimal,misc)
% Optimized upper and lower distance bounds between both compressed data
    % Orthonormal representation in Fourier basis
% Each vector contains complex coefficients 
    % Complex conjugates are excluded (a 2* term is used to account for this, if necessary)
        % We operate on either even length or odd length vectors
        %%
        %% For even: first coefficient (DC) should be divided by sqrt(2) by the user
        %% For odd: user needs to input the DC coefficient as the last entry (as opposed to first one)

% Optimization problem: X = (X^+,X^-), Q = (Q^+,Q^-)
% X^-,Q^- unknown, X^+, Q^+ known
% min (max)      ||X-Q||^2
% s.t.           |X^-_i| <= X^-_max, |Q^-_i| <= Q^-_max
%               \sum_i |X^-_i|^2 = e_X
%               \sum_i |Q^-_i|^2 = e_Q
% e_x, e_q: calculated explicitly from discarded data
% X^-_max:=min_j |X^+_j|, Q^-_max:=min_j |Q^+_j|
%
% Inputs:
% X,Q: vectors to be compressed(NX1)
% numCoeffs: number of coefficients to keep upon compression
% option_algo: First coefficients ('First'), First coefficients + Error ('First_Error')
% Best coefficients ('Best'), Best coefficients + Error ('Best_Error')
% Optimal ('Optimal')
% First: first numCoeffs coefficients stored
% Best: store numCoeffs coefficients of higher magnitude (minimize reconstruction error)
%_Error: Use Cauchy-Schwartz (or product of errors) in calculating upper and lower bounds
%-----------Best_Error is deprecated--------------            
% without this option only one value possible (lb=ub)
% Optimal: Optimization problem described above
% algo_optimal: Algorithm for optimization problem above
% 'Numerical', 'Double_waterfilling', 'Double_waterfilling_old'
%
% Outputs:
% lb: lower bound
% ub: upper bound
% When no information is available on the compression error, only one value is obtained lb=ub

if strcmp(option_algo,'Optimal') && strcmp(algo_optimal,'Numerical')
    [lb,ub] = numerical_cvx(X,Q,numCoeffs,misc);
    return;
end

N = length(X);% = length(Q)

% For avoiding to store complex conjugates
    %% see lines 5-9 of the description
X(end) = X(end)/sqrt(2);
Q(end) = Q(end)/sqrt(2);
power_X = abs(X);%magnitudes of coefficients
power_Q = abs(Q);%magnitudes of coefficients

%bestCoeffs: coefficients stored, otherCoeffs: coefficients discarded
if strcmp(option_algo,'First') || strcmp(option_algo,'First_Error')
    bestCoeffs_X  = 1:numCoeffs;
    otherCoeffs_X = numCoeffs+1:N;
    bestCoeffs_Q  = 1:numCoeffs;%same as for X
    otherCoeffs_Q = numCoeffs+1:N;
else%if strcmp(option_algo,'Best') || strcmp(option_algo,'Best_Error') || strcmp(option_algo,'Optimal')
    [p_X,ind_X] = sort(power_X, 'descend');%#ok<ASGLU> %p_X is dummy variable
    bestCoeffs_X  = ind_X(1:numCoeffs);
    otherCoeffs_X = ind_X(numCoeffs+1:N);
    [p_Q,ind_Q] = sort(power_Q, 'descend');%#ok<ASGLU> %p_Q is dummy variable
    bestCoeffs_Q  = ind_Q(1:numCoeffs);
    otherCoeffs_Q = ind_Q(numCoeffs+1:N);
end

if 1 %strcmp(option_algo,'First_Error') || strcmp(option_algo,'Best_Error') || strcmp(option_algo,'Optimal')
    e_x = norm(X(otherCoeffs_X))^2;% =  norm(X(P1))^2 + norm(X(P3))^2
    e_q = norm(Q(otherCoeffs_Q))^2; %= norm(Q(P2))^2 + norm(Q(P3))^2
    %||X-Q||^2 = ||X||^2 + ||Q||^2 - 2*<X,Q>
    P0 = intersect(bestCoeffs_X,bestCoeffs_Q);%coefficients where X_i,Q_i known
    P1 = intersect(otherCoeffs_X,bestCoeffs_Q);%coefficients where Q_i known, X_i unknown
    P2 = intersect(bestCoeffs_X,otherCoeffs_Q);%coefficients where X_i known, Q_i unknown
    P3 = intersect(otherCoeffs_X,otherCoeffs_Q);%coefficients where X_i,Q_i unknown
    switch option_algo
        case {'First'}%All coefficients are the same
            lb = sqrt(2)*norm(X(P0) - Q(P0));%sqrt(2) needed for storing only one conjugate pair in Fourier representation
            ub = lb;
        case {'Best'}
            lb = sqrt(2)*sqrt(norm(X(P0) - Q(P0))^2 + norm(X(P2))^2 + norm(Q(P1))^2); 
            ub = lb;
        case {'First_Error', 'Best_Error', 'Optimal'}%norms of both vectors are known
            if ~isempty(P0) 
                %P0
                dist = 2*( norm(X)^2 + norm(Q)^2 - 2*real(X(P0)'*Q(P0)));%2* needed for Fourier
            else
                dist = 2*( norm(X)^2 + norm(Q)^2);
            end
            if strcmp(option_algo, 'Optimal')%Double waterfilling or Double_waterfilling_old
                X_max       = min(power_X(bestCoeffs_X));
                Q_max       = min(power_Q(bestCoeffs_Q));
                if strcmp(algo_optimal,'Double_waterfilling_old')
                    res = double_waterfill_old(power_X(P2),power_Q(P1),e_x,e_q,X_max,Q_max);
                elseif strcmp(algo_optimal,'Double_waterfilling')
                    res = double_waterfill(power_X(P2),power_Q(P1),e_x,e_q,X_max,Q_max,P1,P2,P3);
                end
            else
                    %Energies of coefficients that are not common
                e_x = e_x + norm(X(P2))^2;
                e_q = e_q + norm(Q(P1))^2;
                res = sqrt(e_x*e_q);
            end
            
            % *2 for expansion in quadratic
            lb = sqrt(dist - 4*res);
            ub = sqrt(dist + 4*res);
    end
end