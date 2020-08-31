function data = make_unit_norm(data)
% Makes row-wise unit norms of matrix data 

    % make mean zero - to be done in runme
    % m = mean(data,2); % row-wise mean
    % data = (data - repmat(m, 1, size(data,2))); 
    
    scale = vecnorm(data, 2, 2);
    data = data ./ repmat(scale,1,size(data,2)); 
end