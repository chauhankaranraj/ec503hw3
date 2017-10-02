function [LDAmodel] = LDA_train(X_train, Y_train, numofClass)
%
% Training LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%


%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%

% determine how many training points and how many features for each point
[num_data_pts, D] = size(X_train);

% initialize return model parameters
LDAmodel.Mu = zeros(numofClass, D);
LDAmodel.Sigmapooled = zeros(D, D);
LDAmodel.Pi = zeros(numofClass, 1);


%%%%%%%%%%%%%%%%%%%%% CALCULATE CLASS MEAN, PI VECTORS %%%%%%%%%%%%%%%%%%%%

for this_class = 1:numofClass  
    
    % get ONLY data points that are labelled this_class
    this_class_data_pts = X_train((Y_train == this_class), :);
    
    % prior probability is given by num_class_pts/total_pts
    LDAmodel.Pi(this_class, 1) = size(this_class_data_pts, 1) / num_data_pts;
    
    % row-wise mean (mean of each dimension) of all data points of this_class
    LDAmodel.Mu(this_class, :) = mean(this_class_data_pts, 1);
    
end


%%%%%%%%%%%%%%%%%%%%%%% CALCULATE COVARIANCE MATRIX %%%%%%%%%%%%%%%%%%%%%%%

for data_pt_idx = 1:num_data_pts
    
    % calculate (xi-uj)
    temp = X_train(data_pt_idx, :)' - LDAmodel.Mu(Y_train(data_pt_idx, 1), :)';
    
    % add (xi-uj)(xi-uj)' to covariance matrix
    LDAmodel.Sigmapooled = LDAmodel.Sigmapooled + (temp * temp');
    
end

% divide sum by n, the total number of data pts
LDAmodel.Sigmapooled = LDAmodel.Sigmapooled/num_data_pts;

end
