% here for testing purposes
X = rand(15,4);
Y = [2 2 2 2 1 2 1 1 1 2 2 2 1 1 1];
num_classes = 2;
% disp(X);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN-TEST SPLIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% percent of data to be divided into training, validation, and testing set
num_data_pts = size(X, 1);
train_ratio = 2/3;
test_ratio = 1/3;
val_ratio = 0;

% select indices uniformly at random for train and test sets
[train_idx, val_idx, test_idx] = dividerand(num_data_pts, train_ratio, 0, test_ratio);

% get the data at train and test indices
X_train = X(train_idx, :);
X_test = X(test_idx, :);
% disp(X_train);
% disp(X_test);

Y_train = Y(train_idx);
Y_test = Y(test_idx);
% disp(Y_train);
% disp(Y_test);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN CLASSIFIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[my_lda_model] = LDA_train(X_train, Y_train, num_classes);














