function fx = getNormalFFT(x)
%==================================================
% Return normalized Fourier coefficients
%==================================================
    Nx = length(x);
    fx = fft(x)/sqrt(Nx);
