function [QDAmodel]= QDA_train(X_train, Y_train, numofClass)
%
% Training QDA
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
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i


%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%

% get the number of data points in training set, and dimensions of data
[num_data_pts, D] = size(X_train);

% initialize model variables
QDAmodel.Mu = zeros(numofClass, D);
QDAmodel.Sigma = zeros(D, D, numofClass);
QDAmodel.Pi = zeros(numofClass, 1);

% transpose once, use multiple times
X_train_transpose = X_train';


%%%%%%%%%%%%%%%%% CALCULATE CLASS MEAN, PRIOR PROB VECTORS %%%%%%%%%%%%%%%%

for this_class = 1:numofClass
    
    % boolean vector, 1 if data pt belongs to this class and 0 otherwise
    is_data_pt_this_class = (Y_train == this_class);
    num_this_class_data_pts = sum(is_data_pt_this_class);
    
    % update prior prob vector with prio prob estimate of this class
    QDAmodel.Pi(this_class) = num_this_class_data_pts/num_data_pts;
    
    if num_this_class_data_pts
        % get sum of each dimension, ONLY for data pts from this class
        % then divide by total num of data pts from this class
        % then transpose it to get class mean vector as a row
        QDAmodel.Mu(this_class,:) = ((X_train_transpose * is_data_pt_this_class)/num_this_class_data_pts)';
    
    
        % todo: matrix multiply instead of for loop?
    
    end
    

end


%%%%%%%%%%%%%%%% CALCULATE COVARIANCE MATRICES FOR CLASSES %%%%%%%%%%%%%%%%

for this_class = 1:numofClasses
    
end
