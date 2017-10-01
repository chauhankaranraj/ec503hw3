function [Y_predict] = QDA_test(X_test, QDAmodel, numofClass)
%
% Testing for QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test


%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%

% get the number of data points in test set, and dimensions of data
num_data_pts = size(X_test, 1);

% initialize best score to infinity
best_score = realmax;

% initialize predictions vector to 0s
Y_predict = zeros(num_data_pts, 1);


%%%%%%%%%%%%%%%%%%%%% CHECK EACH CLASS FOR EACH DATA PT %%%%%%%%%%%%%%%%%%%

for data_pt_idx = 1:num_data_pts
    
    for test_class = 1:numofClass
        
        % save x-u as temp variable instead of calculating twice
        temp = (X_test(test_class,:)'- QDAmodel.Mu(test_class,:)');
        
        % class depenedent quadratic
        class_quad = 0.5 * (temp * inv(QDAmodel.Sigma(:,:,test_class)) * temp');
        
        % scalar offset
        class_offset = 0.5*ln(det(QDAmodel.Sigma(:,:,test_class))) - ln(QDAmodel.Pi(test_class));
        
        % check if this is the closest match
        if (class_quad + class_offset) < best_score
            best_score = class_quad + class_offset;
            Y_predict(data_pt_idx) = test_class;
        end
        
    end
    
end

end
