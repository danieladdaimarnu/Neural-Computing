% ************************************************************************
%                   TIME & SPACE COMPLEXITY ANALYSIS
% ************************************************************************

% This script performs a Complexity Analysis of the SVM. Time Complexity 
% will be measured varying training sizes. Space Complexity will be measured
% using a the best SVM and varying number of parameters. While testing
% against one of these variables, the others are kept fixed at lower numbers
% to keep computations manageable. 

%%
function SVMSpaceTimeComplexity(data)

    % **************************
    % TIME COMPLEXITY ANALYSIS *
    % **************************

    % Using best hyper-parameter values 
    
    % Data Processing
    input = table2array(normalize(data(:, 1:end-1)));
    target = table2array(data(:, end));

    m = size(input,1); % Nb Rows
    n = size(input,2) + 1; % Nb Columns

    classnames = {'1', '2'};
    trainTimes = [];

    for inputSize = 1000:4000:20000
        % Define Input/Output
        x = input(1:inputSize, :);
        y = target(1:inputSize, :);
    
        % Train Model / Record Training Time
        tic;
        t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 80, 'KernelScale', 1);
        rbfMdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
        toc
        trainTimes = [trainTimes toc];

    end

    % Display Time Complexity Curve
    figure(1);
    plot(trainTimes);
    xlabel('Nb of Training Examples');
    ylabel('Training Time (s)');    
    title('Time Complexity Analysis : SVM RBF w/ KernelScale of 1');

    %%
    % ***************************
    % SPACE COMPLEXITY ANALYSIS *
    % ***************************

    % Kernel Comparison
    classnames = {'1', '2'};
    kernels = {'linear', 'rbf', 'polynomial'};
    storageSizes = [];

    for i = 1:3
        % Define Input/Output
        x = input(1:20000, :);
        y = target(1:20000, :);   
    
        % Train Model / Record Training Time
        if isequal(kernels{i}, 'linear')
            t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 1);
            Mdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);   
        
        elseif isequal(kernels{i}, 'rbf')
            t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 1, 'KernelScale', 1);
            Mdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
        
        else
            t = templateSVM('KernelFunction', 'polynomial', 'BoxConstraint', 1, 'PolynomialOrder', 2);
            Mdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
        end 
    
        modelInfo = whos('Mdl');
        storageSizes = [storageSizes modelInfo.bytes];
    end

    % Display Space Complexity Curve
    figure(2);
    bar(categorical(kernels), storageSizes);
    title('Space Complexity Analysis : SVM training on 20k Examples');
    xlabel('Kernel Types');
    ylabel('Storage Space (Bytes)');

    %% Kernel Scale Comparison
    classnames = {'1', '2'};
    storageSizes = [];

    for kernelscale = 1:4
        % Define Input/Output
        x = input(1:20000, :);
        y = target(1:20000, :);
    
        % Train Model / Record Training Time
        t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 80, 'KernelScale', kernelscale);
        Mdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
        
        modelInfo = whos('Mdl');
        storageSizes = [storageSizes modelInfo.bytes];

    end

    figure(3);
    plot(1:4, storageSizes);
    title('Space Complexity Analysis : SVM training on 20k Examples');
    xlabel('Kernel Scale');
    ylabel('Storage Space (Bytes)');

    %% Box Constraint Comparison
    classnames = {'1', '2'};
    box = [0.1, 1, 10, 30, 50, 60, 80, 100];
    storageSizes = [];

    for i = 1:8
        % Define Input/Output
        x = input(1:20000, :);
        y = target(1:20000, :);
    
        % Train Model / Record Training Time
        t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', box(i), 'KernelScale', 1);
        Mdl = fitcecoc(x, y, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
        
        modelInfo = whos('Mdl');
        storageSizes = [storageSizes modelInfo.bytes];

    end

    figure(4);
    plot(box, storageSizes);
    title('Space Complexity Analysis : SVM training on 20k Examples');
    xlabel('Box Constraint');
    ylabel('Storage Space (Bytes)');

end