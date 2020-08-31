function [x, error_progression] = DLS_new(l,u, I, score, iter,TOL)
% Random projections using upper and lower bounds
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
[m,n] = size(I);
x = zeros(n, 1);

%%%Average of estimates
% Average is given av_x(n) := by 1/(n+1) sum_{k=0}^{n} x(k)
% It can be computed recursively as av_x(n+1) = 1/(n+2)* [(n+1)*av_x(n) + x(n+1)]
av_x = x;

%calculate column probability
p2 = sum(I.^2, 1);
p2 = full(p2);
p2 = p2/sum(p2);
s = randsample(n,iter,true,p2); % Pick nodes according to degrees, s is a vector contains #max_it nodes pick. (line.9) 


fprintf('Start of DLS algorithm, # of iterations: %d\n', iter);
error_progression = zeros(1, iter);

for cnt = 1 : iter
    
    % Display some info every 1000 iterations
     if mod(cnt, 1000) == 0, fprintf('DLS, iter # = %d/ %d)\n', cnt, iter); end
     %%%%%error = norm(I*x - proj_l_u(I*x,l,u)); %%% only works if problem is feasible
     %
     %error = norm(I'* (I*x - proj_l_u(I*x,l,u)) );
     % Error for termination criterion
     error = (1/sqrt(n))* norm(I'* (I*av_x - proj_l_u(I*av_x,l,u)) );
     error_progression(cnt) = error;
     if mod(cnt, 1000) == 0 &&  error < TOL %norm(I*x-(y-z))/(norm(I,'fro')*norm(x)) < TOL && norm(I'*z)/(norm(I,'fro')^2*norm(x))< TOL
        fprintf('DLS stopped after %d iterations\n', cnt);
        error_progression = error_progression(1:cnt);
        break;
     end

    edge_index = randsample(find(I(:,s(cnt))),1); % Pick an edge adjacent to selected node s uniformly at random (line. 10)
    % I_node = full(I(:, s(cnt)));
    I_edge = full(I(edge_index, :));
    
    %Step-size
    gamma = 1;
    %gamma = 1/cnt;
    %gamma = 1/sqrt(cnt);
    %gamma = 1/4;
    %gamma = 1/8;
    
    x = x - gamma * (1 / norm( I_edge)^2 )* (I_edge*x - proj_l_u(I_edge*x,l(edge_index),u(edge_index)))*I_edge';
    
    % Averaged estimates
    av_x = (1/(cnt+1)) * (cnt*av_x + x);
    
end
%x = [x(1:(minInd-1));0;x(minInd:end)];
x = [av_x(1:(minInd-1));0;av_x(minInd:end)];
fprintf('End of DLS algorithm\n');

