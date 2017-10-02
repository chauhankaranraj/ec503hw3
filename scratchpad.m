% num_features = 3;
% numOfClass = 3;
% 
% % labels
% Y_train = [1 2 3 2 2 2 2 1 3 2];
% 
% % 10 rows, 3 cols
% X_train = rand(10,3);
% disp(X_train);
% 
% % num_features x numOfClass matrix, i-th column is mean vector of i-th class
% class_means = zeros(num_features, numOfClass);
% 
% for this_class = 1:numOfClass
%     
%     % num_data_points x 1 boolean vector, with 1 where class label is same as this_class
%     is_pt_this_class = (Y_train == this_class)';
%     num_this_class_pts = sum(is_pt_this_class);
%     
% %     disp(this_class);
% %     disp(is_pt_this_class);
%     
%     if num_this_class_pts    
%         % num_features x 1 vector, where i-th row is sum of feature #i of all data points
%         this_class_feat_sum = (X_train') * is_pt_this_class;
%         
%         % divide feature sum by number of data_points
%         class_means(:,this_class) = this_class_feat_sum/num_this_class_pts;
%     else
%         % no data points for this class, sum will be 0
%         class_means(:,this_class) = zeros(num_features,1);
%     end
%     
% end
% 
% disp(class_means);


% labels = [1 2 1 1 2]';

% test = (train(labels(:)==1,:));
% disp(test);
% 
% mean_vec = mean(test, 1);
% disp(mean_vec);

% data = rand(10,4);
% disp(data);

% train_indices = randi([1 5], 1, 3);
% disp(train_indices);

% X_train = data(train_indices,:);
% X_train = datasample(data, 7, 'Replace', false);
% disp(X_train);


% function ret = scratchpad(num)
% ret = num + 1;
% end

% A = [0 1 1; 2 3 2];
% B = mean(A, 1);
% disp(B);

foo = [1 2 3 4 5 4 10 12 -10 -2]';
disp(foo);

[val, idx] = min(foo( foo>min(foo)));
[val0, idx0] = min(foo);

if idx0 <= idx
    idx = idx + 1;
end

disp(val);
disp(idx);



