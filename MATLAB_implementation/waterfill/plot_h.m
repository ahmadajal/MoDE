%Plot functions h_a, h_b and h
close all; clc; clear;
addpath ./data
A = load('weblog_100x1024.txt');
[n, N] = size(A);
p = randperm(n);
A = A(p,:);
x = A(1,:);
q = A(2,:);
%Normalization
x = (x - mean(x))/std(x);
q = (q - mean(q))/std(q);
fx = getNormalFFT(x);
fq = getNormalFFT(q);
%Remove DC, keep first half of them including the middle one
X = fx(2:(length(x)/2 +1))';    Q = fq(2:(length(q)/2 +1))';
numCoeffs = 12;

N = length(X);% = length(Q)
%For DC in Fourier representation (otherwise unnecessary)
X(end) = X(end)/sqrt(2);
Q(end) = Q(end)/sqrt(2);
power_X = abs(X);
power_Q = abs(Q);
[p_X,ind_X] = sort(power_X, 'descend');%#ok<ASGLU> %p_X is dummy variable
bestCoeffs_X  = ind_X(1:numCoeffs);
otherCoeffs_X = ind_X(numCoeffs+1:N);
[p_Q,ind_Q] = sort(power_Q, 'descend');%#ok<ASGLU> %p_Q is dummy variable
bestCoeffs_Q  = ind_Q(1:numCoeffs);
otherCoeffs_Q = ind_Q(numCoeffs+1:N);
e_x = norm(X(otherCoeffs_X))^2;% =  norm(X(P1))^2 + norm(X(P3))^2
e_q = norm(Q(otherCoeffs_Q))^2; %= norm(Q(P2))^2 + norm(Q(P3))^2
%||X-Q||^2 = ||X||^2 + ||Q||^2 - 2*<X,Q>
P0 = intersect(bestCoeffs_X,bestCoeffs_Q);%coefficients where X_i,Q_i known
P1 = intersect(otherCoeffs_X,bestCoeffs_Q);%coefficients where Q_i known, X_i unknown
P2 = intersect(bestCoeffs_X,otherCoeffs_Q);%coefficients where X_i known, Q_i unknown
P3 = intersect(otherCoeffs_X,otherCoeffs_Q);%coefficients where X_i,Q_i unknown
A = min(power_X(bestCoeffs_X));
B = min(power_Q(bestCoeffs_Q));
a = power_X(P2);
b = power_Q(P1);
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
    v2 = zeros(P2_card,1);
    for i=1:P2_card
        v2(i) = sum(min(a.^2 / g2(i), B^2*ones(P2_card,1))) - e_q;
    end
    if P2_card*B^2 <= e_q;
        gamma_a = 0;
    else
        %v2 (decreasing order)
        i = max(find(v2>=0));
        if v2(i) == 0
            gamma_a = g2(i);
        elseif v2(end) > 0
            gamma_a = e_q / sum(a.^2);
        else
            gamma_a = lin_root(g2(i),g2(i+1),v2(i),v2(i+1));
        end   
    end



    %gamma_b
    v1 = zeros(P1_card,1);
    for i=1:P1_card
        v1(i) = sum(min(b.^2 *g1(i), A^2*ones(P1_card,1))) - e_x;
    end
    if P1_card*A^2 <= e_x;
        gamma_b = +Inf;
    else
        %v2 in increasing order
        if v1(i) == 0
            gamma_b = g1(i);
        elseif v1(1) > 0
            gamma_b = e_x/ sum(b.^2);
        else
            gamma_b = lin_root(g1(i-1),g1(i),v1(i-1),v1(i));
        end
    end
g = union(g1,g2);
g = g(find(g>=gamma_a));
g = g(find(g<=gamma_b));
v = zeros(length(g),1);
for i=1:length(g);
    v(i) = h(g(i),a,b,A,B,e_x,e_q);
end

figure;
subplot(2,2,1);
plot(g2,v2,'LineWidth',1.5);
xlabel('\gamma','fontsize',10);
ylabel('h_a(\gamma)','fontsize',10);
title('Function h_a(\gamma)','fontsize',14);

subplot(2,2,2);
plot(g1,v1,'LineWidth',1.5);
xlabel('\gamma','fontsize',10);
ylabel('h_b(\gamma)','fontsize',10);
title('Function h_b(\gamma)','fontsize',14);

subplot(2,2,3);
plot(g,v,'LineWidth',1.5);
xlabel('\gamma','fontsize',10);
ylabel('h(\gamma)','fontsize',10);
title('Function h(\gamma)','fontsize',14);

subplot(2,2,4);
plot(g,v-g,'LineWidth',1.5);
xlabel('\gamma','fontsize',10);
ylabel('h(\gamma)-\gamma','fontsize',10);
title('Function h(\gamma) - \gamma','fontsize',14);

