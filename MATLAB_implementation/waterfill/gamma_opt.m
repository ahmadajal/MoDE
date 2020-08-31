function gamma = gamma_opt(a,b,A,B,e_x,e_q)
%Calculation of the root of h (as in paper)
%Inputs:
    %a,b: vectors of known coefficients (in P2,P1)
    %A,B: upper bounds on unknown coefficients
    %e_x,e_q: energy of unknown coefficients
%    
%Output:
    %gamma: root of h

        %Cardinalities of P1, P2
b = b(:);
a = a(:);
P1_card = length(b);
P2_card = length(a);

a = sort(a,'ascend');
b = sort(b,'descend');
    %Points of non-differentiability of numerator (increasing order)
g1 = A^2*ones(P1_card,1) ./ (b.^2);
    %Points of non-differentiability of denominator (increasing order)
g2 = a.^2 / B^2;


%Define gamma_a, gamma_b as in paper
    %gamma_a
if P2_card*B^2 <= e_q;
    gamma_a = 0;
else
    v2 = zeros(P2_card,1);
    for i=1:P2_card
        v2(i) = sum(min(a.^2 / g2(i), B^2*ones(P2_card,1))) - e_q;
    end
    %v2 (decreasing order)
    i = max(find(v2>=0));
%     if v2(i) == 0 
%         gamma_a = g2(i);
%     else
%         gamma_a = (e_q - (P2_card - i)*B^2) / sum(a(1:i).^2);
%     end
    if v2(i) == 0 
        gamma_a = g2(i);
    elseif v2(end) > 0
        gamma_a = e_q / sum(a.^2);        
    else
        gamma_a = lin_root(g2(i),g2(i+1),v2(i),v2(i+1));
    end

end
% if gamma_a >0
%    sum(min(a.^2 / gamma_a, B^2*ones(P2_card,1))) - e_q
% end

    %gamma_b
if P1_card*A^2 <= e_x;
    gamma_b = +Inf;
else
    v1 = zeros(P1_card,1);
    for i=1:P1_card
        v1(i) = sum(min(b.^2 *g1(i), A^2*ones(P1_card,1))) - e_x;
    end
    %v1 in increasing order
    i = min(find(v1>=0));
%     if v1(i) == 0 
%         gamma_b = g1(i);
%     else
%         gamma_b = (e_x - (i-1)*A^2) / sum(b(i:P1_card).^2);
%     end
    if v1(i) == 0 
        gamma_b = g1(i);
    elseif v1(1) > 0
        gamma_b = e_x/ sum(b.^2);
    else
        gamma_b = lin_root(g1(i-1),g1(i),v1(i-1),v1(i));
    end
end
 %if gamma_b <+Inf
 %   sum(min(b.^2 *gamma_b, A^2*ones(P1_card,1))) - e_x
 %end
    %Exclude points below gamma_a and above gamma_b
g = union(g1,g2);
g = g(find(g>=gamma_a));
g = g(find(g<=gamma_b));
v = zeros(length(g),1);
for i=1:length(g);
    v(i) = h(g(i),a,b,A,B,e_x,e_q);
end

if isempty(v)
    %   Remove comment for an implementation consistent with our theory but
    %   potentially less stable
    %%Not both gamma_a = 0, gamma_b = +Inf
    %if gamma_a >0%root above points in g
    %    gamma = (e_x +sum(a.^2)-P1_card*A^2) / e_q;
    %else%root below points in g
    %    gamma = e_x / (e_q + sum(b.^2) - P2_card*B^2);
    %end
    gamma=-1;
    return;
end

%Need some rounding for better stability

%Case 1: root at one of the points in g
%   Rounding for stability
accur = 10^-4;
ind = find(accur*round(v/accur)==0);
%ind = find(v==0);
if ~isempty(ind)
    gamma = g(ind(1));
    return;
end
%Case 2: root below points in g
if v(1) >0 
    gamma = e_x / (e_q + sum(b.^2) - P2_card*B^2);
    return;
end

%Case 3: root above points in g
if v(end) < 0
    gamma = (e_x +sum(a.^2)-P1_card*A^2) / e_q;
    return;
end

%Case 4: root in between points in g
i = max(find(v<0));
%gamma = lin_root(g(i),g(i+1),v(i),v(i+1));
gamma = lin_root(g(i),g(i+1),h1(g(i),a,b,A,B,e_x,e_q),h1(g(i+1),a,b,A,B,e_x,e_q));
% h(gamma,a,b,A,B,e_x,e_q)
% P11 = find(g1 <= g(i));
% P12 = find(g1 > g(i));
% P21 = find(g2 >= g(i));
% P22 = find(g2 < g(i));
% %gamma = (e_x - length(P11)*A^2 + sum(B^2*P22)) / (e_q - length(P21)*B^2 + sum(A^2*ones(length(P12),1) ./ P12));
% gamma = (e_x - length(P11)*A^2 + sum(a(P22).^2) ) / (e_q - length(P21)*B^2 + sum(b(P12).^2) );

%Root of a linear function on [x0,x1] with values f0,f1 at x0,x1 such that
%f0f1<0
function x = lin_root(x0,x1,f0,f1)
x = x0 - (x1 - x0) / (f1 - f0) * f0;

%Auxiliary function h1 for finding solution to linear equation for the root of h
function res = h1(gamma,a,b,A,B,e_x,e_q)

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

res = -( e_x - S1 ) + ( e_q - S2 ) * gamma;

