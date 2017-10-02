load data_knnSimulation;

% num_data_pts = size(Xtrain, 1);
% num_classes = size(unique(ytrain(:, 1)), 1);

% points on grid
x_vals = -3.5 : 0.1 : 6;
y_vals = -3 : 0.1 : 6.5;

% value of k for k-nearest neighbors
k = 10;

% initialize probability vectors to 0
prob_class_1 = zeros(length(x_vals), length(y_vals));
prob_class_2 = zeros(length(x_vals), length(y_vals));
prob_class_3 = zeros(length(x_vals), length(y_vals));

for x_idx = 1:length(x_vals)
    
    for y_idx = 1:length(y_vals)        
        
        % difference vectors
        temp = Xtrain - [x_vals(1, x_idx) y_vals(1, y_idx)];

        % distances squared of current point from each point
        distances_sq = sum(temp .* temp, 2);

        % get the indices sorted according to distance squared
        [sorted_dist_sq, sorted_idx] = sort(distances_sq, 'ascend');

        % labels of k nearest neighbors of current data point
        neighbors = ytrain(sorted_idx(1:k, 1), 1);

        % probability of being class i = num class i neighbors/k
        prob_class_1(x_idx, y_idx) = sum(neighbors==1) / k;
        prob_class_2(x_idx, y_idx) = sum(neighbors==2) / k;
        prob_class_3(x_idx, y_idx) = sum(neighbors==3) / k;
        
    end
    
end

imagesc([-3.5 6], [-3 6.5], prob_class_3);
colorbar



