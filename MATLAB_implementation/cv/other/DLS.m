function [x] = DLS(y, I, score, iter,T)
% Implementation of Distributed Linear Solver algorithm
% Choose the lowest score point as anchor.
% Reduced incidence matrix & stop condition

%Inputs:
    % y:         Relative measurements
    % I:         Incidence matrix (mxn)
    % score:     Score column (nx1)
    % iter:      Max number of iterations
    % T:       stop condition precision: eps

    
%Output:
    % x: low-dimensional data point output

% Initialization of v and z 
[~,minInd] = min(score); % find the index of lowest score point
I(:,minInd) = [];% reduced incidence matrix
[m,n] = size(I);
x = zeros(n, 1);
z = y;

% Calculate column probability
p2 = sum(I.^2, 1);
p2 = full(p2);
p2 = p2/sum(p2);
s = randsample(n,iter,true,p2); % Pick nodes according to degrees, s is a vector contains node picks for every iteration.


fprintf('Start of DLS algorithm, # of iterations: %d\n', iter);

for cnt = 1 : iter
    
     % Display some info every 1000 iterations
     if mod(cnt, 1000) == 0, fprintf('DLS, iter # = %d/ %d)\n', cnt, iter); end
     if mod(cnt, 1000) == 0 && norm(I*x-(y-z))/(norm(I,'fro')*norm(x)) < T && norm(I'*z)/(norm(I,'fro')^2*norm(x))< T
        fprintf('DLS stopped at %d\n', cnt);
        break;
     end

    edge_index = randsample(find(I(:,s(cnt))),1); 
    % Pick an edge adjacent to selected node s uniformly at random 
    I_node = full(I(:, s(cnt)));
    I_edge = full(I(edge_index, :));
    z = z - (1 / norm( I_node)^2 ) * (I_node' * z) * I_node;
    % update edge variables 
    
    x = x+(1 / norm( I_edge)^2 )*((y(edge_index,1)-z(edge_index,1))-I_edge*x)* I_edge';  
    % update nodal variables. 

    
end
x = [x(1:(minInd-1));0;x(minInd:end)];
fprintf('End of DLS algorithm\n');

