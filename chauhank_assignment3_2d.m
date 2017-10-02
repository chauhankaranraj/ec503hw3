load data_cancer;

% number of classes in dataset
num_classes = size(unique(Y), 1);

% ccr values for different gammas
ccrs = zeros(19, 1);

% index of gamma in ccr values array
gamma_idx = 1;

for curr_gamma = 0.1 : 0.05 : 1

% classes are 0,1. change to 1,2
Y = Y + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN-TEST SPLIT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% percent of data to be divided into training, validation, and testing set
num_data_pts = size(X, 1);
num_train_pts = 150;
num_test_pts = 66;

% select indices uniformly at random for train and test sets
[train_idx, val_idx, test_idx] = dividerand(num_data_pts, num_train_pts/num_data_pts, 0, num_test_pts/num_data_pts);

% get the data at train and test indices
X_train = X(train_idx, :);
Y_train = Y(train_idx);

X_test = X(test_idx, :);
Y_test = Y(test_idx);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN CLASSIFIER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% train the rda model on data
[my_rda_model] = RDA_train(X_train, Y_train, curr_gamma, num_classes);

% get predictions by the trained rda model
my_rda_preds = RDA_test(X_test, my_rda_model, num_classes);

% make labels 0,1 again
Y = Y - 1;
my_rda_preds = my_rda_preds - 1;

% ccr
rda_ccr = (my_rda_preds==Y_test);

fprintf('for gamma=%f, precision=%f\n', curr_gamma, sum(rda_ccr,1) / size(rda_ccr,1));

% store ccr for this gamma in ccrs array
ccrs(gamma_idx, 1) = sum(rda_ccr,1) / size(rda_ccr,1);

gamma_idx = gamma_idx + 1;

end

x = linspace(0.1, 1, 19);
scatter(x, ccrs)

