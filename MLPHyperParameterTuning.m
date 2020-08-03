% ************************************************************************
%                   HYPER-PARAMETER BAYESIAN OPTIMIZATION
% ************************************************************************
% This script performs Bayesian Optimization to search for the best
% combination of values for hyper-parameters. It relies on a function
% called cvLoss defined in the file cvLoss.m in the same directory.

function MLPHyperParameterTuning(data)
    %% Data processing
    input = table2array(normalize(data(:, 1:end-1)));
    target = table2array(data(:, end));   
    targetOHE = dummyvar(target); % Transform Target into Dummy Variables
    m = size(input,1); % Nb Rows
    n = size(input,2) + 1; % Nb Columns
    
    % Split into train and test
    P = 0.75 ; % 75-25 split
    Xtrain = input(1:round(P*m), :);
    yOHEtrain = targetOHE(1:round(P*m), :);
    Xtest = input(round(P*m)+1:end, :);
    yOHEtest = targetOHE(round(P*m)+1:end, :);
    
    % Define a train/validation split
    cv = cvpartition(size(yOHEtrain,1), 'Holdout', 1/4); % Hold-Out validation inside Objective Function

    %% Specify Hyper-Parameters
   
    vars = [optimizableVariable('networkDepth', [1, 3], 'Type', 'integer');
            optimizableVariable('numHiddenNeurons', [1, 50], 'Type', 'integer');
            optimizableVariable('lr', [1e-3 1], 'Transform', 'log');
            optimizableVariable('momentum', [0.8 0.95]);
            optimizableVariable('trainFcn', {'traingda', 'traingdm', 'traingdx', 'trainscg', 'trainoss'}, 'Type', 'categorical');
            optimizableVariable('transferFcn', {'logsig', 'poslin', 'tansig', 'purelin'}, 'Type', 'categorical')];
        
    %% Optimize Hyper-Parameters
    
    minfn = @(T)cvLoss(Xtrain', yOHEtrain', cv, T.networkDepth, T.numHiddenNeurons, T.lr, T.momentum, T.trainFcn,...
            T.transferFcn);
    results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,'AcquisitionFunctionName',...
              'expected-improvement-plus', 'MaxObjectiveEvaluations', 200);
    T = bestPoint(results);

    %% Train best model on full train set

    hiddenLayerSize = ones(1, T.networkDepth) * T.numHiddenNeurons;
    net = patternnet(hiddenLayerSize, char(T.trainFcn));
    net.trainParam.lr = T.lr; % Update Learning Rate (if any)
    net.trainParam.mc = T.momentum; % Update Momentum Constant (if any)
    net.divideMode = 'none'; % Use all data for Training

    for i = 1:T.networkDepth
        net.layers{i}.transferFcn = char(T.transferFcn); % Update Activation Function of Layers
    end

    [net tr] = train(net, Xtrain', yOHEtrain');

    %% Evaluate on Test Set

    yPred = net(Xtest');
    tind = vec2ind(yOHEtest');
    yPredind = vec2ind(yPred);
    classifError = sum(tind ~= yPredind)/numel(tind);
    fprintf("The Classification Accuracy on the Test Set is : %.2f%%\n", (1-classifError)*100)


end