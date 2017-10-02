load data_mnist_train;
load data_mnist_test;

Y_predict = zeros(size(X_test, 1), 1);

for test_pt_idx = 1:size(X_test, 1)
    
    % difference between current data point vector, and other data vectors
    diffs = X_train - X_test(test_pt_idx, :);
    
    % distance squared between current data point and other points
    distances_sq = sum(diffs .* diffs, 2);
    
%     % get the point with 2nd minimum distance (1st min is itself)
%     [closest_dist_sq, closest_idx] = min(distances_sq( distances_sq>min(distances_sq) ));
%     
%     if (test_pt_idx <= closest_idx)
%         closest_idx = closest_idx + 1;
%     end

    [closest_dist_sq, closest_idx] = min(distances_sq);
    
    Y_predict(test_pt_idx, 1) = Y_train(closest_idx, 1);
    
end


ccr = (Y_predict==Y_test);
disp(sum(ccr, 1) / size(ccr, 1));


