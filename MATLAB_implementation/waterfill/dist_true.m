function d = dist_true(X,Q)
%Euclidean distance between X and Q
    %Orthonormal representation (e.g., Fourier)
        %Each vector contains complex coefficients corresponding to different basis functions (e.g., frequencies)
        %Complex conjugates are excluded (a 2* term is used to account for this, if necessary)
        %Last coefficient is only once (1 / sqrt(2) division is applied for this purpose)
X(end) = X(end) / sqrt(2);
Q(end) = Q(end) / sqrt(2);
d = sqrt(2)*norm(X-Q);