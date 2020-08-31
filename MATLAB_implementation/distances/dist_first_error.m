function [LB, UB] = dist_first_error(fx, fy, numCoeffs)
%==================================================
% First numCoeffs + sum of Squares of omitted coeffs
%==================================================
% fx, fy = first complex Fourier coefficients without their conjugates
% numCoeffs = how many to use for compression

    % sum of squares of omitted 
    fxMiddle = fx(end); fyMiddle = fy(end);
    omittedCoeffs = [(numCoeffs+1):(length(fx)-1)];    
    
    eX = sqrt(2*sum(abs(fx(omittedCoeffs)).^2) + abs(fxMiddle).^2);
    eY = sqrt(2*sum(abs(fy(omittedCoeffs)).^2) + abs(fyMiddle).^2);

    % coeffs we kept
    fx = fx(1:numCoeffs);
    fy = fy(1:numCoeffs);

    dist = 2*sum(abs(fx-fy).^2);

    LB = sqrt(dist + (eX-eY)^2);
    UB = sqrt(dist + (eX+eY)^2);