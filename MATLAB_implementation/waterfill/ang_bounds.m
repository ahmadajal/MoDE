function [ang_lb,ang_ub] = ang_bounds(dist_lb,dist_ub,sign,norm_x,norm_y)
%
% Computes upper and lower bounds on angular difference (needed for CDE)
% from upper/lower bounds on distance
% & norms of the two data points 
% (for normalized data these can be left blank; default value = 1) 

% Formula: dist^2 = norm_x^2 + norm_y^2 -2*corr(x,y)*norm_x*norm_y
% where corr(x,y) is the correlation coefficient 
%    
%%%%%%This implies: corr(x,y) = (norm_x^2 + norm_y^2 - dist^2) / (2*norm_x*norm_y)
% For normalized data (norm_x=norm_y=1) this simply becomes: corr(x,y) = 1 - dist^2/2 
                          
% NOTE: CompressiveMining code gives lb,ub on distance (not distance squared)
% which are taken as input 
% This function operates for a pair of compressed points
% a for-loop will be needed for creating matrices l,u needed in DLS2
    
%---------%---------%---------%---------%---------%---------%---------%---------%---------%---------%
%%%%%
%INPUTS:
% dist_lb:   lower bound on distance
% dist_ub:   upper bound on distance
% norm_x:    norm of one point
% norm_y:    norm of other point
% sign:      sign (-1 or +1)
% i.e., the entry s(i) in FIDE code, where i=1,..,m; 
% m the number of edges in the graph
%%%%%    
%OUTPUTS:
% ang_lb:    lower bound on angular difference
% ang_ub:    upper bound on angular difference
%
% Stacking these values to vectors l,u (mx1 vectors) is the input to DLS2
%%

%For normalized data, can call the function as corr_bounds(dist_lb,dist_ub,s,[],[])
    %Normalized data is default choice if no norms are given
if isempty(norm_x)
    norm_x=1; %normalized 
end

if isempty(norm_y)
    norm_y=1; %normalized 
end

if norm_x*norm_y==0
    error('Non-zero points required'); % Norm cannot be zero
end

% Bounds on correlation
    % 2x1 vector
corr_bounds = (norm_x^2 + norm_y^2 - [dist_lb;dist_ub].^2) / (2*norm_x*norm_y);

ang_bounds = sort(sign*acos(corr_bounds)); % sort in decreasing order

ang_ub = ang_bounds(1);
ang_lb = ang_bounds(2);
