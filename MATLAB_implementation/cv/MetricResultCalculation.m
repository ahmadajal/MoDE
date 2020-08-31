%function [Rs,Rd,Rc, Rd_full]= MetricResultCalculation(X,X_2d,Score,k,type)
function [Rs,Rd,Rc, Rd_full, R_order]= MetricResultCalculation(X,X_2d,Score,k,type,DM,Spearman_opt)
% This function will output R_s, R_c, R_d for all the algorithms in FIDE
% paper.

%Inputs:
    % X:            Original data in higher dimension (nxd)
    % X_2d:         2d embeddings (nx2)
    % Score:        the corresponding score vector (nx1)
    % k:            usually set up as k_FIDE/2
    % type:         'Gd':The embedding algorithm is based on Euclidean distance
    %               'Gc':The embedding algorithm is based on correlation distance
    % DM:           input the average distance matrix for creating the kNN graph for comparisons
                    % or leave empty to create based on original distances
    % Spearman_opt:  'y': Spearman with y-coordinate
                    %'angle': Spearman with angle

%Output:
    % Rc
    % Rd
    % Rs
    % Rd_full: the average ratio of (Euclidean) new_distances/original_distances [0-1]
    % R_order:  percentage of preserved orders

if isempty(DM)    
    DM = squareform(pdist(X)); % original distances -> original kNNG
end

CDM_orig = squareform(pdist(X,'cosine')); % original correlations

if strcmp(type,'Gd')
    [A,~] = adjmat(DM,k); % this is average kNNG if DM is given / original, else
elseif strcmp(type,'Gc') % we don't use it, 
                         % although we have observed it gives better performance for FIDE
                         % if we want to use it, can give (CM_LB+CM_UB)/2 as input DM
                                % (and use 'Gd' still)                                 
    [A,~] = adjmat(CDM,k);
else
    error('Error: Type option should be Gd or Gc')
end
    
CDM_Vis = squareform(pdist(X_2d,'cosine')); 
Temp = arrayfun(@(n) mean(abs(CDM_orig(n,A(n,:)==1)-CDM_Vis(n,A(n,:)==1))),1:size(A,1));
Rc = 1 - mean(Temp);

DM_Vis = squareform(pdist(X_2d));
if ~isempty(DM)
    DM_orig =  squareform(pdist(X)); % original distances
else %already computed as DM above -- line 26
    DM_orig=DM;
end
Temp = arrayfun(@(n) mean(abs(DM_orig(n,A(n,:)==1)-DM_Vis(n,A(n,:)==1))./(DM_orig(n,A(n,:)==1)+DM_Vis(n,A(n,:)==1))),1:size(A,1));

% trying if ratio is better
% Temp = arrayfun(@(n) mean(DM_Vis(n,A(n,:)==1) ./ DM(n,A(n,:)==1) ),1:size(A,1));
% Temp = arrayfun(@(n) mean(abs(DM(n,A(n,:)==1)-DM_Vis(n,A(n,:)==1))./2),1:size(A,1));
Rd = 1 - mean(Temp);

if nargin == 6
    Spearman_opt = 'y';
end
    
switch Spearman_opt
    case 'y'
        Temp = arrayfun(@(n) corr(X_2d(A(n,:)~=0,2),Score(A(n,:)~=0),'Type','Spearman'),1:size(A,1));
        Temp(isnan(Temp)) = 1;
        Rs = mean(Temp);
    case 'angle'
        angle = cart2pol(X_2d(:,1),X_2d(:,2)); % returns angular values in [-pi,pi]
        %%%
        %% Leave next line out--so as to not wrap small negative angles to large positive ones
            %% makes more sense in our LS formulation that does not take mod 2pi %%
        %%    
        % angle = angle + 2*pi*(angle<0); % convert to [0,2pi] 
        %%%
        Temp = arrayfun(@(n) corr(angle(A(n,:)~=0),Score(A(n,:)~=0),'Type','Spearman'),1:size(A,1));
        Temp(isnan(Temp)) = 1;
        Rs = mean(Temp);
    otherwise
        error('Bad option for Spearman');
end

%% Make in [0,1]
%Rs = (1+Rs)/2;

% set diagonals to one, so that division is possible
DM_Vis(logical(eye(size(DM_Vis)))) = 1;
DM_orig(logical(eye(size(DM)))) = 1;

ratios = DM_Vis ./ DM_orig;
Rd_full = mean(ratios(:));


%%%%%%%% Preservation of partial orders %%%%%%%%%%%
%
angle = cart2pol(X_2d(:,1),X_2d(:,2)); % returns angular values in [-pi,pi]
R_order = 0;
n = size(A,1);
edges=0;
for i=1:n
    for j=i+1:n
        if A(i,j)==1 % neighbors in graph
            edges = edges+1;
            if ( (Score(i)>Score(j)) && (angle(i) < angle(j)) ) || ( (Score(j)>Score(i)) && (angle(j) < angle(i)))
                R_order = R_order+1; % cost 1 if order is not preserved
            end
        end                
    end
end
R_order = 1 -  R_order / edges;
%
%%%%%%%% End of Preservation of partial orders %%%%%%%%%%%