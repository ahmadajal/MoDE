function [res,x,E,l,mu] = waterfill(a,e_x,A)
%Water-filling algorithm to solve the optimization problem 
%max    a'x
%s.t.   \sum_i x_i^2 <= e_x      (1)
%       |x_i| <=A                (2)
%
%For the case NA^2>=e_x (l>0)
    %otherwise solution is x = A*ones(N,1)
    %l = 0, mu = a / A
%Inputs:
    %a: vector of known coefficients (Nx1) (non-negative wlog)
    %e_x: energy of error
    %A: upper bound on unknown coefficients
%Outputs:
    %res: optimal value
    %x: optimal solution (Nx1)
    %E: unutilized energy
    %l: Lagrange multiplier for (1)
    %mu: Lagrange multipliers (Nx1) for (2)

N = length(a);    
if N == 0
    res = 0;
    E = e_x;
    x = [];
    l = 0;
    mu = [];
    return;
end

a = abs(a);% in case a is not non-negative or complex
if N*A^2 <= e_x
    x = A*ones(N,1);
    l = 0;
    mu = a/A;
    E = e_x - N*A^2;
    res = a'*x;
else%Waterfilling algorithm
    %Sorting is not necessary, but gives the right order in which coefficients are saturated (i.e., are set equal to A)
    %%[a, ind] = sort(a, 'descend');
    l = 0;
    R = e_x;%energy reserve
    C = 1:N;%non-saturated (i.e., < A) coefficients 
    x = zeros(N,1);
    while (R > 0) && ~isempty(C)%(min(x)<A)
            %original problem
        %l =  1/2* norm(a(C)) / sqrt(R); 
        %x(C) = a(C) / (2*l);
            %sqrt-parametrized problem
        l =  norm(a(C)) / sqrt(R);
        x(C) = a(C) / l;
        i = find(x(C)>=A);
        if isempty(i)
            break;
        end
        x(C(i)) = A;
        C(i) = [];
        %R = e_x -  norm(x(setdiff(C0,C)))^2;
        R = e_x -  (N-length(C))*A^2;
    end

    res = a'*x;
        %original problem
    %mu = a - 2*l*A;
        %sqrt-parametrized problem
    mu = a/A-l;
    mu(C) = 0;
    E = 0;
end
