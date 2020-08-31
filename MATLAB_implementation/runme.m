% This script runs MoDE along with the other 3 methods mentioned in the paper (ISOMAP, MDS and t-SNE).
% For each of the methods the metrics R_d, R_c and R_o are computed and the resulting 2D embeddings
% are plotted.
%close all;
%clear;clc;
disp('Running...');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options (parameters for the embedding methods)
options.NumCoeffs = 4; % (only for MoDE) The number of fft coefficient that are being kept after compression
                      % Could be any number from 0 to ceil((d+1)/2) -- d is the dimension of the data

options.knn = 20; % (for MoDE and ISOMAP) #NNs for kNNG
options.perplexity = 6.5; % (only for t-SNE) perplexity is related to the number of nearest neighbors
                          % that is used in other manifold learning algorithms. Its mathematical relation
                          % #NNs is :
                          %       #NNs = min(n - 1, int(3. * perplexity + 1))
options.MaxIter = 100000; % Max number of iterations for gradient method
options.Precision = 10^(-4); % (only for MoDE) Tolerance or termination criterion for gradient method
options.use_same_LB_UB = 0; % (only for MoDE) Use midpoint as both lb/ub
options.compression = 0; % 0: original data is embedded
                         % 1: compressed data is embedded

options.dims = 1:2; % (for ISOMAP) dimensionality of ISOMAP embeddings
options.verbose = 0; % for ISOMAP
% options.dataset = 'big_stock'; % small_stock or big_stock
options.randomize_scores = 0; % (for MoDE) 0: use the original scores
                              % 1: randomize the scores
%options.Rseval = 'y'; % for Rs evaluation
options.Rseval = 'angle';
options.data = "small_stock.mat"; % dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Experiment for the following options:');
disp(options)

addpath('./distances') % processing
addpath('./waterfill') % bounds
addpath('./cv') % our algo
addpath(genpath('./baselines')) %isomap, lle, etc
addpath('./data') % datasets

tStart = tic; % start timer

dataframe = load(join(["data/", options.data], ""));
data = dataframe.StockData; % data features

score = dataframe.Score;  % scores

if (options.randomize_scores)
    score = randperm(length(score))'; % randomize score to check quality
end

% %% prefereably the data should be already normalised. If not, you can also use
% %% the following normalisation. However, note that the following is more appropriate
% %% for time-series .
%% subtract the mean
m = mean(data,2); % row-wise mean
data = data - repmat(m, 1, size(data,2));
% 
%% Normalize
s = max(data, [], 2) - min(data, [], 2); % 1 x num_objects
data = data ./ repmat(s,1, size(data,2));
%%

% original L2 pairwise distances for IsoMap
D_orig = L2_distance2(data', data',1); % use L2_distance in absence of
                                       % statistics toolbox (pdist)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MoDE
[X_2d, error_progression, DM_average] =  CV2(data,score,options.knn, ...
                                                options.MaxIter, ...
                                                options.Precision, ...
                                                options.NumCoeffs, ...
                                                options.use_same_LB_UB, ...
                                                options.compression);
%% computing metrics
[Rs, Rd, Rc, Rd_full, R_order] = MetricResultCalculation(data,X_2d,score,options.knn,'Gd', D_orig, options.Rseval);

% Definition of function
% plot_metrics(Rc, Rd, Rd_full, Rs, R_order, method_name)
plot_metrics(Rc, Rd, Rd_full, Rs, R_order, 'CV');
plot_data(X_2d, score, "CV", Rd, R_order, Rc, options.data);
%%%%%%% save
data_name = split(options.data, ".");
data_name = data_name(1);
% save(join(["../data/", data_name, "_mode"], ""), "X_2d");
% save(join(["../data/", data_name, "_DM"], ""), "DM_average");

% Convergence plot
figure; semilogy(error_progression);
xlabel('Iterations');
ylabel('Error');
title('Convergence of DAE');

%%% Test other algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ISOMAP
disp('--------------------');
fprintf('ISOMAP on mid-bound distances\n');
[Y, ~, ~] = IsoMap(DM_average, 'k', options.knn, options);
mappedX_ISOM = Y.coords{2,1};
ISOM_2d = mappedX_ISOM';

[Rs,Rd,Rc, Rd_full, R_order] = MetricResultCalculation(data,ISOM_2d,score,options.knn,'Gd', D_orig, options.Rseval);
plot_metrics(Rc, Rd, Rd_full, Rs, R_order, 'ISOMAP');
plot_data(ISOM_2d, score, "ISOMAP", Rd, R_order, Rc, options.data);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MDS
 MDS_2d = cmdscale(DM_average);
 MDS_2d = MDS_2d(:,[1:2]);

[Rs,Rd,Rc, Rd_full, R_order] = MetricResultCalculation(data, MDS_2d,score,options.knn,'Gd',D_orig,options.Rseval);
plot_metrics(Rc, Rd, Rd_full, Rs, R_order, 'MDS');
plot_data(MDS_2d, score, "MDS", Rd, R_order, Rc, options.data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% t-SNE, results are averaged across 10 runs as t-SNE has randomization
t = zeros(10,5);
for i = 1:10

    tSNE_2d = tsne_d(DM_average, [], 2, options.perplexity);
    [Rs, Rd, Rc, Rd_full, R_order] = MetricResultCalculation(data, tSNE_2d, score, options.knn,'Gd',D_orig,options.Rseval);
    t(i,:) = [Rc, Rd, Rd_full, Rs, R_order];
end
metrics_tsne = mean(t);

plot_metrics(metrics_tsne(1), metrics_tsne(2), metrics_tsne(3), metrics_tsne(4), metrics_tsne(5), 't-SNE');
plot_data(tSNE_2d, score, "t-SNE", Rd, R_order, Rc, options.data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tElapsed = toc(tStart);
fprintf('The simulation took %0.3f seconds\n', tElapsed);
