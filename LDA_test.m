function [Y_predict] = LDA_test(X_test, LDAmodel, numofClass)
%
% Testing for LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% LDAmodel : the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test


%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize variables %%%%%%%%%%%%%%%%%%%%%%%%%%

[num_data_pts, num_features] = size(X_test);

% calculate inverse just once, for multiple usages later on
cov_inv = inv(LDAmodel.Sigmapooled);

% initialize return vector
Y_predict = zeros(num_data_pts,1);

% initial "best" score
best_score = realmin;

% todo: check out inverting mechanism

for data_pt = 1:num_data_pts
    
    % find class that maximizes aposteriori probability
    for possible_class = 1:numofClass
        
        % calculate projection onto the class vector and offset
        projection = LDAModel.Mu(possible_class,:)*cov_inv*X_test(data_pt,:)';
        offset = ln(LDAModel.Pi(possible_class)) - 0.5*LDAModel.Mu(possible_class,:)*cov_inv*LDAModel.Mu(possible_class,:)';
        
        % check if it is better than the running best score
        if (projection + offset) > best_score
            best_score = (projection + offset);
            Y_predict(data_pt) = possible_class;
        end
        
    end
    
end

end
