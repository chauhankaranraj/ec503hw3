function [RDAmodel]= RDA_train(X_train, Y_train, gamma, numofClass)
%
% Training RDA
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
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i


%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%

[num_data_pts, D] = size(X_train);

RDAmodel.Mu = zeros(numofClass, D);
RDAmodel.Sigmapooled = zeros(D, D);
RDAmodel.Pi = zeros(numofClass, 1);


%%%%%%%%%%%%%%%%%%%%% CALCULATE CLASS MEAN, PI VECTORS %%%%%%%%%%%%%%%%%%%%

for this_class = 1:numofClass  
    
    % get ONLY data points that are labelled this_class
    this_class_data_pts = X_train((Y_train == this_class), :);
    
    % prior probability is given by num_class_pts/total_pts
    LDAmodel.Pi(this_class, 1) = size(this_class_data_pts, 1) / num_data_pts;
    
    % row-wise mean (mean of each dimension) of all data points of this_class
    LDAmodel.Mu(this_class, :) = mean(this_class_data_pts, 1);
    
end



end
