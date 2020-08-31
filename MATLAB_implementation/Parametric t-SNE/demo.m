%DEMO Demonstration of parametric t-SNE

    train_data = 'arrow_train.mat';
    test_data = 'arrow_test.mat';
    % Load MNIST dataset
    data_train = load(join(['data/', train_data], ""));
    data_test = load(join(['data/', test_data], ""));
    
    train_X = data_train.StockData;
    train_labels = data_train.labels;
%     %% subtract the mean
%     m = mean(train_X,2); % row-wise mean
%     train_X = train_X - repmat(m, 1, size(train_X,2));
%     %% Normalize
%     s = max(train_X, [], 2) - min(train_X, [], 2); % 1 x num_objects
%     train_X = train_X ./ repmat(s,1, size(train_X,2));
    fprintf('number of training data points: %d \n', size(train_X, 1));
    
    test_X = data_test.StockData;
    test_labels = data_test.labels;
%     %% subtract the mean
%     m = mean(test_X,2); % row-wise mean
%     test_X = test_X - repmat(m, 1, size(test_X,2));
%     %% Normalize
%     s = max(test_X, [], 2) - min(test_X, [], 2); % 1 x num_objects
%     test_X = test_X ./ repmat(s,1, size(test_X,2));
    
    
    % Set perplexity and network structure
    perplexity = 6.5;
    layers = [500 500 2000 2];
    
    % Train the parametric t-SNE network
    [network, err] = train_par_tsne(train_X, train_labels, test_X, test_labels, layers, 'CD1');
    
    % Construct training and test embeddings
    mapped_train_X = run_data_through_network(network, train_X);
    mapped_test_X  = run_data_through_network(network, test_X);
    
    % Compute 1-NN error and trustworthiness
    disp(['1-NN error: ' num2str(knn_error(mapped_train_X, train_labels, mapped_test_X, test_labels, 1))]);
    disp(['Trustworthiness: ' num2str(trustworthiness(test_X, mapped_test_X, 12))]);
    
    % Plot test embedding
    figure();
    scatter(mapped_test_X(:,1), mapped_test_X(:,2), 9, test_labels);
    title('Embedding of test data');
    
    % Plot train embedding
    figure();
    scatter(mapped_train_X(:,1), mapped_train_X(:,2), 9, train_labels);
    title('Embedding of train data');
    
    data_name = split(train_data, ".");
    data_name = data_name(1);   
    save(join(["../data/", data_name, "_ptsne"], ""), "mapped_train_X");
    data_name = split(test_data, ".");
    data_name = data_name(1); 
    save(join(["../data/", data_name, "_ptsne"], ""), "mapped_test_X");