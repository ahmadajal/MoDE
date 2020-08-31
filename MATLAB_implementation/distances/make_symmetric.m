function B = make_symmetric(A)
% copy upper triangular part to lower to make symmetric 

if sum(sum(imag(A)))>0
    warning("Warning: in make_symmetric, matrix has imaginary parts....");
end

B=A'+tril(A',1)';
