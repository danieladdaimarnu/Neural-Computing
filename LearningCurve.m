% ************************************************************************
%                             LEARNING CURVE
% ************************************************************************

% This script visualizes the Learning Curve of the MLP and SVM Algorithms. 
% It uses the best configuration from the hyper-parameter tuning. In order 
% to stabilize the results, 10-fold cross validation is used to output 
% accuracy estimates for both training and validation along with their 
% estimated errors as represented by the standard deviation.

function LearningCurve(data)

    % Data Processing
    input = table2array(normalize(data(:, 1:end-1)));
    target = table2array(data(:, end));
    targetOHE = dummyvar(target); % Transform Target into Dummy Variables

    %% Create Learning Curve
    MLP_trainAvgScores = [];
    MLP_trainStdScores = [];
    MLP_valAvgScores = [];
    MLP_valStdScores = [];

    SVM_trainAvgScores = [];
    SVM_trainStdScores = [];
    SVM_valAvgScores = [];
    SVM_valStdScores = [];

    % Parameters
    % MLP
    netSize = 34;
    trainFcn = 'trainscg';
    % SVM
    kernel = 'rbf';
    kernelScale = 1;
    boxConstraint = 80;
    % Experiment
    k_folds = 10;
    inputRange = [1000 5000 10000 15000 20000];
    classnames = {'1', '2'};

    for inputSize = inputRange
        % Print Iteration Info
        fprintf("\n- Input Range %d \n", inputSize);
    
        % Define Input/Output
        idx = randperm(numel(target), inputSize);
        x = input(idx, :)';
        y = target(idx, :);
        t = targetOHE(idx, :)';
    
        % Create Cross Validation Folds (Stratified)
        cv = cvpartition(target(idx), 'KFold', k_folds, 'Stratify', true);
    
        % Iterate over the Partitions and fit the model (Cross Val)
        MLP_trainScores = [];
        MLP_valScores = [];
        SVM_trainScores = [];
        SVM_valScores = [];
        for iter = 1:k_folds
            % Print Iteration Info
            fprintf("    Cross Validation Fold %d/%d \n", iter, k_folds);
        
            % Define Model
                % MLP
            net = feedforwardnet(netSize, trainFcn);
            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'softmax';
            net.performFcn = 'crossentropy';
            net.plotFcns = {'plotperform'};
            rng = 1:cv.NumObservations;
            net.divideFcn = 'divideind'; % Manually define train/test sets
            net.divideParam.trainInd = rng(cv.training(iter));
            net.divideParam.testInd = rng(cv.test(iter));
                % SVM
            SVM_Template = templateSVM('KernelFunction', kernel, 'BoxConstraint', boxConstraint, 'KernelScale', kernelScale);
            
            % Train Model
                % MLP
            [net tr] = train(net, x, t);
                % SVM
            svm = fitcecoc(x(:, cv.training(iter))', y(cv.training(iter), :), 'Learners', SVM_Template, 'Coding', 'onevsone', 'ClassNames', classnames);
            
            % Get Train/Validation Scores (Accuracy)
                % MLP
            MLP_pred = net(x);
            tind = vec2ind(t);
            yind = vec2ind(MLP_pred);
            MLP_trainScore = sum(tind(cv.training(iter)) == yind(cv.training(iter)))/numel(tind(cv.training(iter)));
            MLP_valScore = sum(tind(cv.test(iter)) == yind(cv.test(iter)))/numel(tind(cv.test(iter)));
                % SVM
            SVM_pred = str2double(predict(svm, x'));
            SVM_trainScore = sum(SVM_pred(cv.training(iter))==y(cv.training(iter)))/length(SVM_pred(cv.training(iter)));
            SVM_valScore = sum(SVM_pred(cv.test(iter))==y(cv.test(iter))) / length(SVM_pred(cv.test(iter)));
            
            % Append Single Iteration Scores
                % MLP
            MLP_trainScores = [MLP_trainScores MLP_trainScore];
            MLP_valScores = [MLP_valScores MLP_valScore];
                % SVM
            SVM_trainScores = [SVM_trainScores SVM_trainScore];
            SVM_valScores = [SVM_valScores SVM_valScore];            
        end
    
        % Append Average/Standard Deviation of Cross Validation Scores
            % MLP
        MLP_trainAvgScores = [MLP_trainAvgScores mean(MLP_trainScores)];
        MLP_trainStdScores = [MLP_trainStdScores std(MLP_trainScores)];
        MLP_valAvgScores = [MLP_valAvgScores mean(MLP_valScores)];
        MLP_valStdScores = [MLP_valStdScores std(MLP_valScores)];
            % SVM
        SVM_trainAvgScores = [SVM_trainAvgScores mean(SVM_trainScores)];
        SVM_trainStdScores = [SVM_trainStdScores std(SVM_trainScores)];
        SVM_valAvgScores = [SVM_valAvgScores mean(SVM_valScores)];
        SVM_valStdScores = [SVM_valStdScores std(SVM_valScores)];        
    end


    %% Visualize Results
        % MLP
    figure(1);
    patch([inputRange fliplr(inputRange)], [MLP_trainAvgScores+MLP_trainStdScores, fliplr(MLP_trainAvgScores-MLP_trainStdScores)], [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputRange fliplr(inputRange)], [MLP_valAvgScores+MLP_valStdScores, fliplr(MLP_valAvgScores-MLP_valStdScores)], [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(inputRange, MLP_trainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputRange, MLP_valAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - MLP');
    legend('Train Score Error', 'Validation Score Error', 'Train Score Estimate', 'Validation Score Estimate');

        % SVM
    figure(2);
    patch([inputRange fliplr(inputRange)], [SVM_trainAvgScores+SVM_trainStdScores, fliplr(SVM_trainAvgScores-SVM_trainStdScores)], [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([inputRange fliplr(inputRange)], [SVM_valAvgScores+SVM_valStdScores, fliplr(SVM_valAvgScores-SVM_valStdScores)], [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(inputRange, SVM_trainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(inputRange, SVM_valAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('Learning Curve - SVM');
    legend('Train Score Error', 'Validation Score Error', 'Train Score Estimate', 'Validation Score Estimate');

end
