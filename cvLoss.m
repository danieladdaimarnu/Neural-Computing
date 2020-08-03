% ************************************************************************
%                 OBJECTIVE FUNCTION OF BAYESIAN OPTIMIZATION
% ************************************************************************

% This script defines the Objective Function that will be used in the
% bayesian Optimization procedure for Hyper-Parameter tuning. It builds a
% Neural Network and evaluates its performance on a Holdout set.

% Define Cross-Validation Loss Function
function classifError = cvLoss(x, t, cv, networkDepth, numHiddenNeurons, lr, momentum, trainFcn, transferFcn)

% Build Network Architecture
hiddenLayerSize = numHiddenNeurons * ones(1, networkDepth); % Define vector of Hidden Layer Size (Network Architecture)
net = patternnet(hiddenLayerSize, char(trainFcn)); % Build Network
net.trainParam.epochs = 500; % Specify Nb of epochs
net.trainParam.max_fail = 6; % Early stopping after 6 consecutive increases of Validation Performance
if any(strcmp({'traingda', 'traingdm', 'traingdx'} , char(trainFcn)))
    net.trainParam.lr = lr; % Update Learning Rate
    if ~strcmp('traingda', char(trainFcn))
        net.trainParam.mc = momentum; % Update Momentum Constant
    end
end
for i = 1:networkDepth % Update Activation Function of Layers
    net.layers{i}.transferFcn = char(transferFcn); 
end

% Divide Training Data into Train-Validation sets
rng = 1:cv.NumObservations;
net.divideFcn = 'divideind';
net.divideParam.trainInd = rng(cv.training);
net.divideParam.valInd = rng(cv.test);


% Train Network
net = train(net, x, t);

% Evaluate on validation set and compute Classification Error
y = net(x);
tind = vec2ind(t);
yind = vec2ind(y);
classifError = sum(tind(cv.test) ~= yind(cv.test))/numel(tind(cv.test));

end