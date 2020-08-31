function X_2d = laplacian_eigenmap_2d(D, knn)

sigma2 = 100; % determines strength of connection in graph... 

%% now let's get pairwise distance info and create graph 
m                = size(D,1);
[srtdDt,srtdIdx] = sort(D,'ascend');
D                = srtdDt(1:knn+1,:);
nidx             = srtdIdx(1:knn+1,:);

% nz   = dt(:) > 0;
% mind = min(dt(nz));
% maxd = max(dt(nz));

% compute weights
tempW  = exp(-D.^2/sigma2); 

% build weight matrix
i = repmat(1:m,knn+1,1);
W = sparse(i(:),double(nidx(:)),tempW(:),m,m); 
W = max(W,W'); % for undirected graph.

% The original normalized graph Laplacian, non-corrected for density
ld = diag(sum(W,2).^(-1/2));
DO = ld*W*ld;
DO = max(DO,DO');%(DO + DO')/2;

% get eigenvectors
[v,d] = eigs(DO,10,'la');

cmap = jet(m);

eigVecIdx = nchoosek(2:4,2);
for i = 1:size(eigVecIdx,1)
    figure,scatter(v(:,eigVecIdx(i,1)),v(:,eigVecIdx(i,2)),20,cmap)
    title('Nonlinear embedding');
    xlabel(['\phi_',num2str(eigVecIdx(i,1))]);
    ylabel(['\phi_',num2str(eigVecIdx(i,2))]);
end


X_2d = [v(:,3) v(:,4)]

