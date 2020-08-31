function X_2d = mds_2d(D)
% performs Linear Multidimensional Scaling and returns 2D coordinates

% squaring
sq=D'*D;
% double centering
n=size(D,1);
J=eye(n)-(1/n)*ones(n);
G=-1/2*J*sq*J;

%% SVD
[U2,S2,V2]=svd(G,'econ');
P2=G*V2(:,1:2);

%% 2D coordinates
X_2d = P2(:,[1:2]);