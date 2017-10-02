% classes are 0,1. change to 1,2
Y = Y + 1;

% number of classes in dataset
num_classes = size(unique(Y), 1);


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
Y_train = Y(train_idx);

X_test = X(test_idx, :);
Y_test = Y(test_idx);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN CLASSIFIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

curr_gamma = 0.15;

% train the rda model on data
[my_rda_model] = RDA_train(X_train, Y_train, curr_gamma, num_classes);

% get predictions by the trained rda model
my_rda_preds = RDA_test(X_test, my_rda_model, num_classes);

% make labels 0,1 again
Y = Y - 1;
my_rda_preds = my_rda_preds - 1;

% precision
rda_score = (my_rda_preds==Y_test);
disp(sum(rda_score,1) / size(rda_score,1));



