load data_iris;

% number of classes in the data set
num_classes = size(unique(Y), 1);

for split_num = 1:10

%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN-TEST SPLIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% percent of data to be divided into training, validation, and testing set
num_data_pts = size(X, 1);
train_ratio = 2/3;
test_ratio = 1/3;
val_ratio = 0;

% select indices uniformly at random for train and test sets
[train_idx, val_idx, test_idx] = dividerand(num_data_pts, train_ratio, val_ratio, test_ratio);

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

% train the lda model on data
[my_lda_model] = LDA_train(X_train, Y_train, num_classes);

% get predictions by the trained lda model
my_lda_preds = LDA_test(X_test, my_lda_model, num_classes);

% precision
lda_score = (my_lda_preds==Y_test);
disp(sum(lda_score,1) / size(lda_score,1));

% train the qda model on data
[my_qda_model] = QDA_train(X_train, Y_train, num_classes);

% get predictions by the trained qda model
my_qda_preds = QDA_test(X_test, my_qda_model, num_classes);

% precision
qda_score = (my_qda_preds==Y_test);
disp(sum(qda_score,1) / size(qda_score,1));

% cp = classperf(Y_test, my_qda_preds);
% disp(cp.CorrectRate);
% classperf(cp, my_qda_preds);
% disp(cp.CorrectRate);

sep = zeros(18,1);
disp(sep');



end


