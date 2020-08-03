% ************************************************************************
%                   TIME & SPACE COMPLEXITY ANALYSIS
% ************************************************************************

% This script performs a Complexity Analysis of the Multi-Layer Perceptron. 
% Time and Space Complexity will be measured using the best MLP with varying 
% numbers of Training Examples, Attributes and Hidden Neurons. While testing
% against one of these variables, the others are kept fixed at lower numbers
% to keep computations manageable. 

%%
function MLPSpaceTimeComplexity(data)
    %% Data Processing 
    input = table2array(normalize(data(:, 1:end-1)));
    target = table2array(data(:, end));
    targetOHE = dummyvar(target); % Transform Target into Dummy Variables


    %% Training Time as a Function of Number of Examples in Training Set
    rowsAvgTrainTimes = []; rowsStdTrainTimes = [];
    nb_experiments = 10;
    inputRowsRange = 1000:4000:20000;

    fprintf('Training Time as a Function of Number of Examples in Training Set.\n')
    for inputRows = inputRowsRange
        rowsTrainTimes = [];
        fprintf('    Input Size : %d \n', inputRows)
        for rep = 1:nb_experiments
            fprintf('        Experiment Nb : %d \n', rep)
            % Define Input/Output
            idx = randperm(20000, inputRows); % Randomize data selection
            x = input(idx, 1:8)';
            t = targetOHE(idx, :)';

            % Define Model
            net = feedforwardnet(34, 'trainscg');
            net.divideMode = 'none'; % Use all data for Training
            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'softmax';
            net.performFcn = 'crossentropy';
            net.plotFcns = {'plotperform'};

            % Train Model / Record Training Time
            [net tr] = train(net, x, t);
            rowsTrainTimes = [rowsTrainTimes tr.time(end)];
             
        end
    
        % Record Average and Standard Deviation of Experiments 
        rowsAvgTrainTimes = [rowsAvgTrainTimes mean(rowsTrainTimes)];
        rowsStdTrainTimes = [rowsStdTrainTimes std(rowsTrainTimes)];
    end

    % Display Time Complexity Curve
    figure(1);
    patch([inputRowsRange fliplr(inputRowsRange)], [rowsAvgTrainTimes+rowsStdTrainTimes, fliplr(rowsAvgTrainTimes-rowsStdTrainTimes)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(inputRowsRange, rowsAvgTrainTimes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel('Nb of Training Examples');
    ylabel('Training Time (s)');
    title('Time Complexity Analysis : Number of Examples');
    legend('Train Time Error', 'Train Time Estimate');


    %% Training Time as a Function of Number of Attributes in Training Set
    colsAvgTrainTimes = []; colsStdTrainTimes = [];
    nb_experiments = 10;
    inputColsRange = 1:2:8;

    fprintf('Training Time as a Function of Number of Attributes in Training Set\n')
    for inputCols = inputColsRange % Iterate over Number_Of_Attributes' Values
        colsTrainTimes = [];
    
        fprintf('    Input Nb Attributes : %d \n', inputCols)
        % Repeat the experiment nb_experiments times
        for rep = 1:nb_experiments
        
            fprintf('        Experiment Nb : %d \n', rep)
            % Define Input/Output
            idx = randperm(20000, 4000); % Randomize data selection
            x = input(idx, 1:inputCols)';
            t = targetOHE(idx, :)';

            % Define Model
            net = feedforwardnet(34, 'trainscg');
            net.divideMode = 'none'; % Use all data for Training
            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'softmax';
            net.performFcn = 'crossentropy';
            net.plotFcns = {'plotperform'};

            % Train Model / Record Training Time
            [net tr] = train(net, x, t);
            colsTrainTimes = [colsTrainTimes tr.time(end)];
        
        end
    
        % Record Average and Standard Deviation of Experiments
        colsAvgTrainTimes = [colsAvgTrainTimes mean(colsTrainTimes)];
        colsStdTrainTimes = [colsStdTrainTimes std(colsTrainTimes)];
    end

    % Display Time Complexity Curve
    figure(2);
    patch([inputColsRange fliplr(inputColsRange)], [colsAvgTrainTimes+colsStdTrainTimes, fliplr(colsAvgTrainTimes-colsStdTrainTimes)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(inputColsRange, colsAvgTrainTimes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel('Nb of Attributes');
    ylabel('Training Time (s)');
    title('Time Complexity Analysis : Number of Attributes');
    legend('Train Time Error', 'Train Time Estimate');

    %% Training Time, Prediction Time and Storage Space as a Function of Neural Network Size (Hidden Neurons)
    netTrainAvgTimes = []; netTrainStdTimes = [];
    storageAvgSizes = []; storageStdSizes = [];
    netPredAvgTimes = []; netPredStdTimes = [];
    nb_experiments = 10;
    netSizeRange = 10:10:100;

    fprintf('Neural Network Size : Train Time, Prediction Time and Storage Space \n')
    for netSize = netSizeRange % Iterate over Network Size Values
        netTrainTimes = [];
        storageSizes = [];
        netPredTimes = [];
    
        fprintf('    Network Size : %d \n', netSize)
        % Repeat the same experiment nb_experiments times
        for rep = 1:nb_experiments
        
            fprintf('        Experiment Nb : %d \n', rep)
            % Define Input/Output (20000 examples, 8 attributes)
            idx = randperm(20000, 4000); % Randomize the selection of train data
            x = input(idx, 1:8)';
            t = targetOHE(idx, :)';

            % Define Model
            net = feedforwardnet(netSize, 'trainscg');
            net.divideMode = 'none'; % Use all data for Training
            net.layers{1}.transferFcn = 'logsig';
            net.layers{2}.transferFcn = 'softmax';
            net.performFcn = 'crossentropy';
            net.plotFcns = {'plotperform'};

            % Train Model
            [net tr] = train(net, x, t);
        
            % Record Training Time & Storage Space
            netTrainTimes = [netTrainTimes tr.time(end)];
            modelInfo = whos('net');
            storageSizes = [storageSizes modelInfo.bytes];

            % Record Prediction Time of a single example
            tic
            pred = net(x(:,1));
            netPredTimes = [netPredTimes toc];     
        end
    
        % Record Average and Standard Deviation of Experiments
        netTrainAvgTimes = [netTrainAvgTimes mean(netTrainTimes)];
        netTrainStdTimes = [netTrainStdTimes std(netTrainTimes)];
        storageAvgSizes = [storageAvgSizes mean(storageSizes)]; 
        storageStdSizes = [storageStdSizes std(storageSizes)];
        netPredAvgTimes = [netPredAvgTimes mean(netPredTimes)];
        netPredStdTimes = [netPredStdTimes std(netPredTimes)];
    end

    % Display Time (Train/Prediction) and Space Complexity Curves
    figure(3); % Training Time
    patch([netSizeRange fliplr(netSizeRange)], [netTrainAvgTimes+netTrainStdTimes, fliplr(netTrainAvgTimes-netTrainStdTimes)],...
        [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(netSizeRange, netTrainAvgTimes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel('Nb of Hidden Neurons');
    ylabel('Training Time (s)');
    title('Train Time Complexity Analysis : Network Size');
    legend('Train Time Error', 'Train Time Estimate');

    figure(4); % Prediction Time
    patch([netSizeRange fliplr(netSizeRange)], [netPredAvgTimes+netPredStdTimes, fliplr(netPredAvgTimes-netPredStdTimes)], [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(netSizeRange, netPredAvgTimes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel('Nb of Hidden Neurons');
    ylabel('Training Time (s)');
    title('Prediction Time Complexity Analysis : Network Size');
    legend('Prediction Time Error', 'Prediction Time Estimate');

    figure(5); % Storage Space
    patch([netSizeRange fliplr(netSizeRange)], [storageAvgSizes+storageStdSizes, fliplr(storageAvgSizes-storageStdSizes)], [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    line(netSizeRange, storageAvgSizes, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    xlabel('Nb of Hidden Neurons');
    ylabel('Storage Space (Bytes)');
    title('Space Complexity Analysis : Network Size');
    legend('Storage Space Error', 'Storage Space Estimate');

    %% Save Important Variables
    %save('../Results/Space Time Complexity/SpaceTimeResults_MLP.mat', 'nb_experiments',...
    %    'inputRowsRange', 'rowsAvgTrainTimes', 'rowsStdTrainTimes', 'inputColsRange', 'colsAvgTrainTimes', ...
    %    'colsStdTrainTimes', 'netSizeRange', 'netTrainAvgTimes', 'netTrainStdTimes', 'storageAvgSizes', ...
    %    'storageStdSizes', 'netPredAvgTimes', 'netPredStdTimes')

end