function [res,a,b] = double_waterfill(a,b,e_x,e_q,A,B,P1,P2,P3)
%Double Water-filling algorithm to solve the optimization problem described
    %in paper

%Inputs:
    %b: vector of known coefficients of b in P1
    %a: vector of known coefficients of a in P2
        %coefficients ordered in P1,P2,P3
    %e_x: energy of error of a
    %A: upper bound on unknown coefficients of a
    %e_q: energy of error of b
    %B: upper bound on unknown coefficients of B
    %P1,P2,P3: subsets of [1,..,N] as defined in paper
%Outputs:
    %res: optimal value

a = a(:);
b = b(:);
p_x_minus = union(P1,P3);%p^-_x
p_q_minus = union(P2,P3);%p^-_q

%Case 1
if isempty(intersect(p_x_minus,p_q_minus))
    if isempty(p_x_minus)%a: all coefficients known P1=P3=O
        res = waterfill(a,e_q,B);
        return;
    end
    if isempty(p_q_minus)%b: all coefficients known P2=P3=O
        res = waterfill(b,e_x,A);
        return;
    end
    if isempty(P3)
        res1 = waterfill(b,e_x,A);
        res2 = waterfill(a,e_q,B);
        res = res1 + res2;
        return;
    end
end   
%Case 2:
if isempty(P1) && isempty(P2)
    res = sqrt(e_x)*sqrt(e_q);
    return;    
end
if (e_x <= length(P1)*A^2) && (e_q <= length(P2)*B^2)
    res1 = waterfill(b,e_x,A);
    res2 = waterfill(a,e_q,B);
    res = res1+res2;
else
    gamma = gamma_opt(a,b,A,B,e_x,e_q);
    if gamma==-1
        res = double_waterfill_old(a,b,e_x,e_q,A,B);
        return;
    end
    %h(gamma,a,b,A,B,e_x,e_q)
    e_x_prime = e_x - sum(min(b.^2*gamma,A^2*ones(length(b),1)));
    e_q_prime = e_q - sum(min(a.^2/gamma,B^2*ones(length(a),1)));
    res1 = waterfill(b,e_x-e_x_prime,A);
    res2 = waterfill(a,e_q-e_q_prime,B);
    res3 = sqrt(e_x_prime)*sqrt(e_q_prime);
    res = res1 + res2 + res3;
end
