load data_mnist_train;
load data_mnist_test;

% knn predictions
Y_predict = zeros(size(X_test, 1), 1);

for test_pt_idx = 1:size(X_test, 1)
    
    % difference between current data point vector, and other data vectors
    diffs = X_train - X_test(test_pt_idx, :);
    
    % distance squared between current data point and other points
    distances_sq = sum(diffs .* diffs, 2);

    % since k=1, select the label of closest point
    [closest_dist_sq, closest_idx] = min(distances_sq);
    
    Y_predict(test_pt_idx, 1) = Y_train(closest_idx, 1);
    
end

% correct classification rate
ccr_mat = (Y_predict==Y_test);
ccr = (sum(ccr_mat, 1) / size(ccr_mat, 1));

% confusion matrix
conf_mat = confusionmat(Y_test, Y_predict);