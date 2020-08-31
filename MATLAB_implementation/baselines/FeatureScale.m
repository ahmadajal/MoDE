function [X_new]= FeatureScale(X)
% Feature scale data using min-max normalization
X_new = (X - min(X)) ./ (max(X) - min(X));