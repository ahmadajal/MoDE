function [x, error_progression] = DLS_SAG_A(l,u, I, score, iter,TOL, Algo, precon, option)
%%%%% SAG & SAGA for the problem%%%%%%
%
% Choose the lowest score point as anchor.
% Reduced incidence matrix & stop condition

%Inputs:
    % [l,u]:     Lower/upper bound on relative measurements
    % I:         Incidence matrix (mxn)
    % score:     Score column (nx1)
    % iter:      Max number of iterations
    % Tol:       stop condition precision: eps
    % Algo:      Options are 'SAG' & 'SAGA' 
    % precon:    Preconditioning, 'Y' or 'N'              
    % option:    Option is 'edge' and 'node'
                 % Split into components represeneting 
                    % edge functions -- (m) or
                    % node functions -- (n)
                 %%%%% ONLY edge implemented for simplicity of expressions
    
%Output:
    % x: low-dimensional data point output

% Initialization 
[~,minInd] = min(score); % find the index of lowest score point
I(:,minInd) = [];% reduced incidence matrix
[m,n] = size(I);
x = zeros(n, 1);

if strcmp(precon,'Y')
    %Pre-conditioning (to cast in projection form)
    D = diag (1 ./ sqrt(sum(I.^2,2)));
    I = D*I;
    l = D*l;
    u = D*u;
end

% Initialization for staled gradient and total gradient
if strcmp(option,'edge')
    % Staled gradients
    s_grad = zeros(n,m);
    for edge=1:m % could vectorize here
        I_edge = full(I(edge, :));
        s_grad(:,edge) = (I_edge*x - proj_l_u(I_edge*x, l(edge),u(edge))) * I_edge';
        %s_grad(:,edge) = (I(edge,:)*x - proj_l_u(I(edge,:)*x, l(edge),u(edge))) * I(edge,:)';
    end
    % Total gradient
    %t_grad = 1/m * I'*(I*x - proj_l_u(I*x,l,u));
    t_grad = 1/m * sum(s_grad,2);
    %
elseif strcmp(option,'node')
    error('Method node not implemented');
else
    error('Option can be either edge or node');
end

%Step-size
    %gamma = 1/n;
    %gamma = 1/(2*n);
    %gamma = 1 / max(eigs(I'*I))
    %gamma = 1/(2*20);%20 is k_FIDE
    %gamma = 2 / (2+n); for node
    %gamma = 2 / (2+m); for edge
    
if strcmp(precon,'Y')    
    gamma = 1; % for preconditioned
elseif strcmp(precon, 'N')
    gamma = 1/2; % for non-preconditioned
end
    
fprintf('Start of DLS algorithm, # of iterations: %d\n', iter);
error_progression = zeros(1, iter);

%Iterations of SAG(A)
for cnt = 1 : iter
    
    % Display some info every 100 iterations
     if mod(cnt, 1000) == 0, fprintf('DLS, iter # = %d/ %d)\n', cnt, iter); end
     
     % Error for termination criterion
     error_norm = (1/sqrt(n))* norm(I'* (I*x - proj_l_u(I*x,l,u)) );
     error_progression(cnt) = error_norm;
     if mod(cnt, 1000) == 0 &&  error_norm < TOL %norm(I*x-(y-z))/(norm(I,'fro')*norm(x)) < TOL && norm(I'*z)/(norm(I,'fro')^2*norm(x))< TOL
        fprintf('DLS stopped after %d iterations\n', cnt);
        error_progression = error_progression(1:cnt);
        break;
     end

     %Select an edge uniformly at random
     edge = randsample(m,1);
     % Evaluate gradient of i-th component
     I_edge = full(I(edge, :));
     new_grad_edge = (I_edge*x - proj_l_u(I_edge*x, l(edge),u(edge))) * I_edge';
     %new_grad_edge = (I(edge,:)*x - proj_l_u(I(edge,:)*x, l(edge),u(edge))) * I(edge,:)';
     % Staled gradient
     old_grad_edge = s_grad(:,edge);
     % Update staled value
     s_grad(:,edge) = new_grad_edge;
     
     if strcmp(Algo, 'SAG')
         % Update total gradient
         t_grad = t_grad + (1/m)* (new_grad_edge - old_grad_edge);
         % Update estimate
         x = x - gamma * t_grad;
     elseif strcmp(Algo, 'SAGA')
         % Difference of new/old gradient
         grad_dif_edge = new_grad_edge - old_grad_edge;
         % Update estimate
         x = x - gamma * (t_grad + grad_dif_edge);
         % Update total gradient
         t_grad = t_grad + (1/m)* grad_dif_edge;
     else
         error('Algo has to be either SAG or SAGA');
     end
     
end

x = [x(1:(minInd-1));0;x(minInd:end)];
fprintf('End of DLS algorithm\n');

