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


%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize variables %%%%%%%%%%%%%%%%%%%%%%%%%%

% determine how many training points and how many features for each point
[num_data_pts, num_features] = size(X_train);

% transpose it once, use it many times
X_train_transpose = X_train';

% initialize return model parameters
LDAmodel.Mu = zeros(numofClass, num_featues);
LDAmodel.Sigmapooled = zeros(num_features);
LDAmodel.Pi = zeros(numofClass,1);


%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate class means %%%%%%%%%%%%%%%%%%%%%%%%%%

% num_features x numOfClass matrix, i-th column is mean vector of i-th class
class_means = zeros(num_features, numofClass);

for this_class = 1:numofClass
    
    % num_data_points x 1 boolean vector, with 1 where class label is same as this_class
    is_pt_this_class = (Y_train == this_class);
    num_this_class_pts = sum(is_pt_this_class);
    
    % prior probability is given by num_class_pts/total_pts
    LDAmodel.Pi(this_class) = num_this_class_pts/num_data_pts;
    
    if num_this_class_pts    
        % num_features x 1 vector, where i-th row is sum of feature #i of all data points
        this_class_feat_sum = (X_train_transpose) * is_pt_this_class;
        
        % divide feature sum by number of data_points
        class_means(:,this_class) = this_class_feat_sum/num_this_class_pts;
    else
        % no data points for this class, sum will be 0
        class_means(:,this_class) = zeros(num_features,1);
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%% Calculate covariance matrix %%%%%%%%%%%%%%%%%%%%%%%

for data_pt_num = 1:num_data_pts
    
    % get label for this data point
    data_pt_class = Y_train(data_pt_num);
    
    % calculate (xi-ui)(xi-ui)' and add the result to covariance matrix
    LDAmodel.Sigmapooled = LDAmodel.Sigmapooled + (X_train(data_pt_num,:)' - class_means(data_pt_class)) * (X_train(data_pt_num,:)' - class_means(data_pt_class)';

end

LDAmodel.Sigmapooled = LDAmodel.Sigmapooled/num_data_pts;

% reassign according to desired output format
LDAmodel.Mu = class_means';

end
