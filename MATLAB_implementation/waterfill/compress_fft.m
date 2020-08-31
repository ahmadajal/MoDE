function data_new = compress_fft(data,NumCoeffs, option)
% compress data using NumCoeffs coefficient
    % FFT -> keep NumCoeffs (rest are zero)-> IFFT
    % option: 'Best' or 'First'

[N,dim]= size(data);   % N is number of points, dim is dimension 
data_new = zeros(N,dim);

if (NumCoeffs>dim) || (NumCoeffs<0) 
    error('NumCoeffs cannot exceed dimension');
end

is_even = ~(mod(dim,2)); % even or odd

for i=1:N
    fx = getNormalFFT(data(i,:)); % fft
    
    if is_even    
        lenGth = dim/2 +1; % ignore symmetric ones
        fx_short = fx(1:lenGth); 
        fx_short(1) = fx_short(1)/sqrt(2); % divide DC by sqrt(2) 
    else %if odd
        lenGth = (dim+1)/2; % ignore symmetric ones
        fx_short = fx(1:lenGth);
        fx_short = [fx_short(2:end), fx_short(1)]; % bring DC at the end 
    end
    
    fx_short(end) = fx_short(end) / sqrt(2); % last one taken once in symmetry of fft
    
    switch option
        case 'First'
            bestCoeffs_X = 1:NumCoeffs;
        case 'Best'
            power_X = abs(fx_short);
            [~,ind_X] = sort(power_X, 'descend');
            bestCoeffs_X  = ind_X(1:NumCoeffs);
    end
    
    fx_new_short = zeros(1,lenGth);
    fx_new_short(bestCoeffs_X) = fx_short(bestCoeffs_X);
    
    if is_even    
        fx_new = [ fx_new_short(1:lenGth), conj(fx_new_short((lenGth-1):-1:2)) ] ;
        fx_new(1) = fx_new(1)*sqrt(2);
        fx_new(lenGth) = fx_new(lenGth)*sqrt(2);
    else %if odd
        fx_new = [sqrt(2)*fx_short(end), fx_short(1:(lenGth-1)), conj(fx_short((lenGth-1):-1:1)) ];
    end
    
    xnew = sqrt(dim)*ifft(fx_new);
    data_new(i,:) = xnew;
end