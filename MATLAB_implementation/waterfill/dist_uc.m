function [lb,ub] = dist_uc(Q,X,numCoeffs,option_algo)
%Optimized upper and lower distance bounds between compressed and uncompressed data
    %Orthonormal representation (e.g., Fourier)
    %Each vector contains complex coefficients corresponding to different basis functions (e.g., frequencies)
    %Complex conjugates are excluded (a 2* term is used to account for this, if necessary)
    %Last coefficient is DC (real) (1 / sqrt(2) division is applied for this purpose, if necessary)
%Optimization problem: X = (X^+,X^-), Q = (Q^+,Q^-)
    %X^- unknown, X^+, Q known
    %min (max)      ||X-Q||^2
    %s.t.           |X^-_i| <= X^-_max
    %               \sum_i |X^-_i|^2 <= e_x
        %e_x: calculated explicitly from discarded data
        %X^-_max:=min_j |X^+_j|
%
%Inputs:
    %Q: known vector (NX1)
    %X: vector to be compressed (NX1)
    %numCoeffs: number of coefficients to keep upon compression
    %option_algo: First coefficients ('First'), First coefficients + Error ('First_Error')
        %Best coefficients ('Best'), Best coefficients + Error ('Best_Error')
        %Optimal via waterfilling('Optimal'), Optimal via cvx('Optimal_cvx'),
            %First: first numCoeffs coefficients stored
            %Best: store numCoeffs coefficients of higher magnitude (minimize reconstruction error)
            %_Error: Use Cauchy-Schwartz in calculating upper and lower bounds
                %without this option only one value possible (lb=ub)
%Outputs:
    %lb: lower bound
    %ub: upper bound
        %When no information is available on the compression error, only one value is obtained lb=ub

N = length(X);% = length(Q);
    %For DC in Fourier representation (otherwise unnecessary)
X(end) = X(end) / sqrt(2);
Q(end) = Q(end) / sqrt(2);

if numCoeffs == N
    lb = sqrt(2)*norm(X-Q);
    ub = lb;
    return;
end


power = abs(X);%magnitudes of coefficients

if strcmp(option_algo,'First') || strcmp(option_algo,'First_Error')
    bestCoeffs  = 1:numCoeffs;%coefficients stored
    otherCoeffs = numCoeffs+1:N;%coefficients discarded
else%if strcmp(option_algo,'Best') || strcmp(option_algo,'Best_Error') || strcmp(option_algo,'Optimal') || strcmp(option_algo,'Optimal_cvx')
    [p,ind] = sort(power, 'descend');%p is dummy variable
    bestCoeffs  = ind(1:numCoeffs);%coefficients stored
    otherCoeffs = ind(numCoeffs+1:N);%coefficients discarded
end

if strcmp(option_algo,'First_Error') || strcmp(option_algo,'Best_Error') || strcmp(option_algo,'Optimal')|| strcmp(option_algo,'Optimal_cvx')
    e_x = norm(X(otherCoeffs))^2;
    e_q = norm(Q(otherCoeffs))^2;
end

%||X-Q||^2 = ||X||^2 + ||Q||^2 - 2*<X,Q>

switch option_algo
    case {'First', 'Best'}
        lb = sqrt(2)* norm(X(bestCoeffs) - Q(bestCoeffs)); %sqrt(2) needed for storing only one conjuagte pair in Fourier representation
        ub = lb;
    case {'First_Error', 'Best_Error', 'Optimal', 'Optimal_cvx'}%norm of both vectors are known
        dist = 2*( norm(X)^2 + norm(Q)^2 - 2*real(X(bestCoeffs)'*Q(bestCoeffs)));%2* needed for Fourier
        %dist = 2*( norm(X(bestCoeffs)-Q(bestCoeffs))^2);%2* needed for Fourier
        if strcmp(option_algo, 'Optimal') || strcmp(option_algo, 'Optimal_cvx')
            X_max = min(power(bestCoeffs));
        end
        if strcmp(option_algo, 'Optimal')
            res = waterfill(abs(Q(otherCoeffs)),e_x,X_max);
        elseif strcmp(option_algo, 'Optimal_cvx')
            cvx_quiet(true);
            cvx_solver SDPT3
            cvx_begin
                variable x(N) complex;
                minimize norm(x-Q);
                subject to
                    x(bestCoeffs) == X(bestCoeffs);
                    abs(x(otherCoeffs)) <= X_max*ones(length(otherCoeffs),1);
                    norm(x(otherCoeffs)) <= norm(X(otherCoeffs));
            cvx_end
            res = 1/4 * (dist - 2*cvx_optval^2);
        else
            res = sqrt(e_x*e_q);%Cauchy-Schwartz inequality. 
        end
        
        % *2 for expansion in quadratic
        lb = sqrt(dist - 4*res);
        ub = sqrt(dist + 4*res);
end


