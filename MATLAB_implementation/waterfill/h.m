function res = h(gamma,a,b,A,B,e_x,e_q)
%Function h (as in paper)
%Inputs:
    %a: vectors of known coefficients in P2
    %b: vectors of known coefficients in P1
    %A,B: upper bounds on unknown coefficients
    %e_x,e_q: energy of unknown coefficients
%    
%Output:
    %res: value of h

if isempty(b)
    S1=0;
else
    S1 = sum(min(b.^2*gamma,A^2*ones(length(b),1)));
end

if isempty(a)
    S2=0;
else
    S2 = sum(min(a.^2/gamma,B^2*ones(length(a),1)));
end

res = -( e_x - S1 ) / ( e_q - S2 ) + gamma;

