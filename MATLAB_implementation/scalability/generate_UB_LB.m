%% This script produces the distance and correlation lower and upper bound matrices.
%% It was used to generate these matrices for a big dataset (EEG eye state). Then
%% the subsets of these matrices were used for the scalability experiment.
%% WARNING: running this scipt may take a while
disp('Running...');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Options
options.NumCoeffs = 4; % Any number from 0 to ceil((d+1)/2) -- d is the dimensionality of the data
options.use_same_LB_UB = 0; % Use midpoint as both lb/ub

options.data = "eeg.mat";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Experiment for the following options:');
disp(options)


tStart = tic; % start timer

dataframe = load(join(["data/", options.data], ""));
data = full(dataframe.StockData); % change sparse vector to dense
score = dataframe.Score;

[DM_avg, DM_LB, DM_UB, CM_LB, CM_UB] = compress(data, options.NumCoeffs, options.use_same_LB_UB);

% save
dataset_name = split(options.data, ".");
save(join(["compressed_dist_matrices/", dataset_name(1), "_DM_avg"], ""), "DM_avg");
save(join(["compressed_dist_matrices/", dataset_name(1), "_DM_LB"], ""), "DM_LB");
save(join(["compressed_dist_matrices/", dataset_name(1), "_DM_UB"], ""), "DM_UB");
save(join(["compressed_dist_matrices/", dataset_name(1), "_CM_LB"], ""), "CM_LB");
save(join(["compressed_dist_matrices/", dataset_name(1), "_CM_UB"], ""), "CM_UB");


tElapsed = toc(tStart);
fprintf('The simulation took %0.3f seconds\n', tElapsed);
