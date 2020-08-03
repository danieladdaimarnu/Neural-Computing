%                             DECISION BOUNDARY
% ************************************************************************

% The purpose of this script is to compare the shape of Decision Boundaries
% resulting from a Multi-Layer Perceptron and SVM. 
%%

function DecisionBoundary(data)
    %% define predictor and labels
    cols = {'mean_ds', 'mean_ip'}; 
    X = table2array(normalize(data(:, cols)));
    y = table2array(data(:, end));
    y1 = dummyvar(y);
   
    %% build grid
    minX1 = min(X(:,1)); maxX1 = max(X(:,1)); step1 = (maxX1-minX1)/400;
    minX2 = min(X(:,2)); maxX2 = max(X(:,2)); step2 = (maxX2-minX2)/400;
    [xx1, xx2] = meshgrid(minX1:step1:maxX1, minX2:step2:maxX2);
    XGrid = [xx1(:), xx2(:)];
    
    %% Classifiers
    rng('default') % for reproducibility
   
    %SVM classifier
    % define svm model
    C = 80; % Misclassification Cost
    kernel = 'rbf';
    kernelScale = 1; % Controls the Gamma parameter when the kernel is Gaussian
    % train svm classifier
    mdlSVM = fitcsvm(X, y, 'KernelFunction', kernel, 'BoxConstraint', C, 'KernelScale', kernelScale);
    % SVM predictions
    [predSVM, predProbSVM] = predict(mdlSVM, XGrid); % Predict Class Labels & Posterior Probabilities
    
    
    % MLP classifier.
    % define MLP model
    netSize = 34;
    trainFcn = 'trainscg';
    netModel = feedforwardnet(netSize, trainFcn);
    netModel.layers{1}.transferFcn = 'logsig';
    netModel.layers{2}.transferFcn = 'softmax';
    netModel.performFcn = 'crossentropy';
    netModel.divideParam.trainRatio = .75;
    netModel.divideParam.valRatio = .15;
    netModel.divideParam.testRatio = .1;
    netModel.plotFcns = {'plotperform', 'plotconfusion'};
    % train MLP classifier
    trainedNet = train(netModel, X', y1');
    % MLP predictions
    predProbMLP = trainedNet(XGrid'); % Predicted Scores
    predMLP = vec2ind(predProbMLP); % Predicted Classes
    
    %% Visualize Decision Surface
    figure('pos', [100 300 1000 400])
    % SVM
    subplot(1,2,1)
    scatter(xx1(predSVM==1), xx2(predSVM==1), 1, [100 149 237]/255, 'MarkerEdgeAlpha', .5)
    hold on;
    scatter(xx1(predSVM==2), xx2(predSVM==2), 1, [205 92 92]/255, 'MarkerEdgeAlpha', .5)
    scatter(X(y==1, 1), X(y==1, 2), 20, [100 149 237]/255, 'filled');
    scatter(X(y==2, 1), X(y==2, 2), 20, [205 92 92]/255, 'filled');
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    legend('Prediction Region 1', 'Prediction Region 2','negative', 'positive');
    xlabel('x'); ylabel('y');
    title('SVM');
    hold off;
    
    % MLP
    subplot(1,2,2)
    scatter(xx1(predMLP==1), xx2(predMLP==1), 1, [100 149 237]/255, 'MarkerEdgeAlpha', .5)
    hold on;
    scatter(xx1(predMLP==2), xx2(predMLP==2), 1, [205 92 92]/255, 'MarkerEdgeAlpha', .5)
    scatter(X(y==1, 1), X(y==1, 2), 20, [100 149 237]/255, 'filled');
    scatter(X(y==2, 1), X(y==2, 2), 20, [205 92 92]/255, 'filled');
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    legend('Prediction Region 1', 'Prediction Region 2','negative', 'positive');
    xlabel('x'); ylabel('y');
    title('MLP');
    suptitle('Decision Surface of Classes');
    hold off;
    
    %% Visualize Decision Probabilities (2D)
    
    figure('pos', [100 300 1000 400])
    % SVM
    subplot(1,2,1)
    surf(xx1, xx2, reshape(predProbSVM(:,1), size(xx1)), 'EdgeColor', 'none')
    hold on;
    surf(xx1, xx2, reshape(predProbSVM(:,2), size(xx2)), 'EdgeColor', 'none')
    view(2);
    colormap(parula)
    colorbar
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    title('SVM');
    xlabel('x'); ylabel('y');
    hold off;
    
    % MLP
    subplot(1,2,2)
    surf(xx1, xx2, reshape(predProbMLP(1,:), size(xx1)), 'EdgeColor', 'none')
    hold on;
    surf(xx1, xx2, reshape(predProbMLP(2,:), size(xx1)), 'EdgeColor', 'none')
    view(2);
    colormap(parula)
    colorbar
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    xlabel('x'); ylabel('y');
    title('MLP');
    suptitle('Decision Probabilities (2D)');
    hold off;
    
    %% Visualize Class Probabilities (3D)
    figure('pos', [100 300 1000 400])
    % SVM
    subplot(1,2,1)
    surf(xx1, xx2, reshape(predProbSVM(:,1), size(xx1)), 'FaceColor', [100 149 237]/255, 'EdgeColor', 'none')
    hold on;
    surf(xx1, xx2, reshape(predProbSVM(:,2), size(xx1)), 'FaceColor', [205 92 92]/255, 'EdgeColor', 'none')
    alpha(.4);
    legend('negative', 'positive');
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    xlabel('x'); ylabel('y'); zlabel('Class Probability');
    title('SVM');
    hold off;
    
    % MLP
    subplot(1,2,2)
    surf(xx1, xx2, reshape(predProbMLP(1,:), size(xx1)), 'FaceColor', [100 149 237]/255, 'EdgeColor', 'none')
    hold on;
    surf(xx1, xx2, reshape(predProbMLP(2,:), size(xx1)), 'FaceColor', [205 92 92]/255, 'EdgeColor', 'none')
    alpha(.4);
    legend('negative', 'positive');
    xlim([minX1, maxX1]);
    ylim([minX2, maxX2]);
    xlabel('x'); ylabel('y'); zlabel('Class Probability');
    title('MLP');
    suptitle('Class Probabilities (3D)');
    hold off;

end