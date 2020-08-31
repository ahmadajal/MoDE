function [x, error_progression] = DLS_gradient(l,u, I, score, iter,TOL, k)
%%%%% Gradient method for LS problem %%%%%%
%
% Choose the lowest score point as anchor.
% Reduced incidence matrix & stop condition

%Inputs:
    % [l,u]:     Lower/upper bound on relative measurements
    % I:         Incidence matrix (mxn)
    % score:     Score column (nx1)
    % iter:      Max number of iterations
    % Tol:       stop condition precision: eps

    
%Output:
    % x: low-dimensional data point output

% Initialization 
[~,minInd] = min(score); % find the index of lowest score point
I(:,minInd) = [];% reduced incidence matrix
[~,n] = size(I);
x = zeros(n, 1);

fprintf('Start of DLS algorithm, # of iterations: %d\n', iter);
error_progression = zeros(1, iter);

%Step-size
    %gamma = 1/n;
    %gamma = 1/(2*n);
    %gamma = 1 / max(eigs(I'*I))
    gamma = 1/(3*k); 
    %gamma = 1/(2*k);
    gamma = 1 / (2* max(diag(I'*I)));
tic;    
for cnt = 1 : iter
    % Display some info every some iterations
     if mod(cnt, 50000) == 0, fprintf('DLS, iter # = %d/ %d)\n', cnt, iter); end
     %%%%%error = norm(I*x - proj_l_u(I*x,l,u)); %%% only works if problem is feasible
     %
     %error = norm(I'* (I*x - proj_l_u(I*x,l,u)) );
     %
     % Error for termination criterion -- RMSE value
     error = (1/sqrt(n))* norm(I'* (I*x - proj_l_u(I*x,l,u)) );
     error_progression(cnt) = error;
     if mod(cnt, 1000) == 0 &&  error < TOL %norm(I*x-(y-z))/(norm(I,'fro')*norm(x)) < TOL && norm(I'*z)/(norm(I,'fro')^2*norm(x))< TOL
        fprintf('DLS stopped after %d iterations\n', cnt);
        error_progression = error_progression(1:cnt);
        break;
     end

    x = x - gamma * (I'* (I*x - proj_l_u(I*x,l,u)));
    
end
x = [x(1:(minInd-1));0;x(minInd:end)];
fprintf('End of DAE algorithm\n');
fprintf('The solution took %0.3f sec\n',toc);

