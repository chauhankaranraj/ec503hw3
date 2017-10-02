load data_knnSimulation;

% scatter plot of points from different groups in different colors
figure;
gscatter(Xtrain(:, 1), Xtrain(:, 2), ytrain(:,1))
xlabel('feature 1');
ylabel('feature 2');