function [lb,ub] = numerical_cvx(X,Q,numCoeffs, misc)
%Optimized upper and lower distance bounds between both compressed data
    %using numerical convex optimization via cvx
%Orthonormal representation (e.g., Fourier)
    %Each vector contains complex coefficients corresponding to different basis functions (e.g., frequencies)
    %Complex conjugates are excluded (a 2* term is used to account for this, if necessary)
   %Last coefficient is only once (1 / sqrt(2) division is applied for this purpose)
%Optimization problem: X = (X^+,X^-), Q = (Q^+,Q^-)
%X^-,Q^- unknown, X^+, Q^+ known
%min (max)      ||X-Q||^2
%s.t.           |X^-_i| <= X^-_max, |Q^-_i| <= Q^-_max
%               \sum_i |X^-_i|^2 <= e_x   
%               \sum_i |Q^-_i|^2 <= e_q  
%
    %e_x, e_q: calculated explicitly from discarded data
    %X^-_max:=min_j |X^+_j|, Q^-_max:=min_j |Q^+_j|
%
%Inputs:
    %X,Q: vectors to be compressed(NX1)
    %numCoeffs: number of coefficients to keep upon compression
%Outputs:
    %lb: lower bound
    %ub: upper bound

N = length(X);% = length(Q)
    %For DC in Fourier representation (otherwise unnecessary)
X(end) = X(end)/sqrt(2);
Q(end) = Q(end)/sqrt(2);

if numCoeffs == N%No compression
    lb = sqrt(2)*norm(X-Q);
    ub=lb;
    return;
end
X_abs = abs(X);%magnitudes of coefficients of X
Q_abs = abs(Q);%magnitudes of coefficients of Q

[p_X,ind_X] = sort(X_abs, 'descend');%#ok<ASGLU> %p_X is dummy variable
bestCoeffs_X  = ind_X(1:numCoeffs);
otherCoeffs_X = ind_X(numCoeffs+1:N);
[p_Q,ind_Q] = sort(Q_abs, 'descend');%#ok<ASGLU> %p_Q is dummy variable
bestCoeffs_Q  = ind_Q(1:numCoeffs);
otherCoeffs_Q = ind_Q(numCoeffs+1:N);

X_max = min(X_abs(bestCoeffs_X));
Q_max = min(Q_abs(bestCoeffs_Q));

% 
%Minimization problem
if (misc)    
    cvx_begin quiet
        cvx_solver SDPT3;
        cvx_precision best;

        variable q(N,1) complex;
        variable r(N,1) complex;
        minimize norm(q-r)
        subject to
            q(bestCoeffs_X) == X(bestCoeffs_X);
            r(bestCoeffs_Q) == Q(bestCoeffs_Q);
            abs(q(otherCoeffs_X)) <= X_max*ones(length(otherCoeffs_X),1);
            abs(r(otherCoeffs_Q)) <= Q_max*ones(length(otherCoeffs_Q),1);
            norm(q(otherCoeffs_X)) <= norm(X(otherCoeffs_X));
            norm(r(otherCoeffs_Q)) <= norm(Q(otherCoeffs_Q));
    cvx_end
else
    cvx_begin quiet
        cvx_solver SDPT3;
        
        variable q(N,1) complex;
        variable r(N,1) complex;
        minimize norm(q-r)
        subject to
            q(bestCoeffs_X) == X(bestCoeffs_X);
            r(bestCoeffs_Q) == Q(bestCoeffs_Q);
            abs(q(otherCoeffs_X)) <= X_max*ones(length(otherCoeffs_X),1);
            abs(r(otherCoeffs_Q)) <= Q_max*ones(length(otherCoeffs_Q),1);
            norm(q(otherCoeffs_X)) <= norm(X(otherCoeffs_X));
            norm(r(otherCoeffs_Q)) <= norm(Q(otherCoeffs_Q));
    cvx_end
end;
lb = 2*cvx_optval^2;
P0 = intersect(bestCoeffs_X,bestCoeffs_Q);
dist = 2*( norm(X)^2 + norm(Q)^2 - 2*real(X(P0)'*Q(P0)));
v = dist - lb;
ub = dist + v;

lb = sqrt(lb);
ub=sqrt(ub);
