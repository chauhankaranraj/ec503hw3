function [Y_predict] = RDA_test(X_test, RDAmodel, numofClass)
%
% Testing for RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix 
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test


%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%

num_data_pts = size(X_test, 1);

% calculate inverse just once, for multiple usages later on
cov_inv = inv(RDAmodel.Sigmapooled);

% initialize return vector
Y_predict = zeros(num_data_pts, 1);


%%%%%%%%%%%%%%%%%%%%% CHECK EACH DATA PT FOR EACH CLASS %%%%%%%%%%%%%%%%%%%

for data_pt_idx = 1:num_data_pts
    
    % initial best score
    best_score = realmin;
    
    % find class that maximizes aposteriori probability
    for test_class = 1:numofClass
        
        % calculate projection onto the class vector and offset
        projection = RDAmodel.Mu(test_class,:) * cov_inv * X_test(data_pt_idx,:)';
        offset = log(RDAmodel.Pi(test_class,1)) - 0.5*RDAmodel.Mu(test_class,:)*cov_inv*RDAmodel.Mu(test_class,:)';
        
        % check if it is better than the running best score
        if (projection + offset) > best_score
            best_score = (projection + offset);
            Y_predict(data_pt_idx, 1) = test_class;
        end
                
    end
    
end

end
