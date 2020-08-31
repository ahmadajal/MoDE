function [res,a,b] = double_waterfill_old(a,b,e_x,e_q,A,B)
%Greedy Double Water-filling algorithm to solve the optimization problem described
    %in paper

%Inputs:
    %b: vector of known coefficients of b in P1
    %a: vector of known coefficients of a in P2
        %coefficients ordered in P1,P2,P3
    %e_x: energy of error of a
    %A: upper bound on unknown coefficients of a
    %e_q: energy of error of b
    %B: upper bound on unknown coefficients of B
%Outputs:
    %res: optimal value
    %a: optimal vector (unknown coefficients of a in P1,P3 + known coefficients of a in P2)
    %b: optimal vector (unknown coefficients of b in P2,P3 + known coefficients of b in P1)

[res1,dummy1,e_x_prime] = waterfill(b,e_x,A);
[res2,dummy2,e_q_prime] = waterfill(a,e_q,B);
res3 = sqrt(e_x_prime*e_q_prime);
res = res1+res2+res3;
