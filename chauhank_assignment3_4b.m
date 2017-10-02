load data_knnSimulation;

num_data_pts = size(Xtrain, 1);
num_classes = size(unique(ytrain(:, 1)), 1);

% value of k for k-nearest neighbors
k = 10;

% initialize probability vectors to 0
prob_class_1 = zeros(num_data_pts, 1);
prob_class_2 = zeros(num_data_pts, 1);
prob_class_3 = zeros(num_data_pts, 1);

for data_pt_idx = 1:size(Xtrain, 1)
    
    % difference vectors
    temp = Xtrain - Xtrain(data_pt_idx, :);
    
    % distances squared of current point from each point
    distances_sq = sum(temp .* temp, 2);
    
    % get the indices sorted according to distance squared
    [sorted_dist_sq, sort_idx] = sort(distances_sq, 'ascend');
    
    % k nearest neighbors of current data point
    neighbors = ytrain(sort_idx(2:k+1, 1), 1);

    % probability of being class i = num class i neighbors/k
    prob_class_1(data_pt_idx, 1) = sum(neighbors==1) / k;
    prob_class_2(data_pt_idx, 1) = sum(neighbors==2) / k;
    prob_class_3(data_pt_idx, 1) = sum(neighbors==3) / k;

end

contour(tri, Xtrain(:,1), Xtrain(:,2), prob_class_2)
