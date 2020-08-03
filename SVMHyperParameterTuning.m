%  HyperParameterTuning uses the 'OptimizeHyperparameters' name-value pair 
%  argument to find optimal parameters automatically. 

function SVMHyperParameterTuning(data)
    %% Data processing
    Xtrain = table2array(normalize(data(:, 1:end-1)));
    Ytrain = table2array(data(:, end));
    
    %% Linear SVM
    % Note the wide range of values in parameters tested in Grid Search here, which will be 
    % followed by a narrowed search later on with the best model.
    fprintf('Start of Linear svm:\n\n')
    rng(88);
    classnames = {'1', '2'};
    C = [0.1, 1, 10, 50];

    training_acc_lin = [];
    val_acc_lin = [];

    % 5-fold CV partition for train and validation sets
    idx = randperm(numel(Ytrain), size(Ytrain, 1));
    x = Xtrain(idx, :);
    y = Ytrain(idx, :);
    cv1 = cvpartition(Ytrain(idx), 'KFold', 5, 'Stratify', true);

    for j = 1:4
        lin_trainScores = [];
        lin_valScores = [];
    
        % Iteration of each fold in 5-fold CV partition
        for iter = 1:4
            % Model training
            t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', C(j));
            LinMdl = fitcecoc(x(cv1.training(iter), :), y(cv1.training(iter), :), 'Learners', t, 'Coding', 'onevsone', ...
            'ClassNames', classnames);
             disp('Mdl done!')
        
            % Get Train/Validation Scores (Accuracy)
            pred = str2double(predict(LinMdl, x));
            trainScore = sum(pred(cv1.training(iter))==y(cv1.training(iter)))/length(pred(cv1.training(iter)));
            disp('Training done!')
            disp(trainScore)
            valScore = sum(pred(cv1.test(iter))==y(cv1.test(iter))) / length(pred(cv1.test(iter)));
            disp('Validation done!')
            disp(valScore)
        
            % Append Single Iteration Scores
            lin_trainScores = [lin_trainScores trainScore];
            lin_valScores = [lin_valScores valScore];
            disp('Prediction done!')
            
        end
        disp(j)
    
        % Append Average of Cross Validation Scores
        training_acc_lin = [training_acc_lin mean(lin_trainScores)];
        val_acc_lin = [val_acc_lin mean(lin_valScores)];
    end
    
    %writematrix(val_acc_lin, 'val_acc_lin.csv')
    %writematrix(training_acc_lin, 'training_acc_lin.csv')
    fprintf('End of Linear svm.\n\n')
    
    %% SVM with Kernel Trick
    % Not using Crossval for SVM with kernel trick due to computation limitation
    fprintf('Start of RBF svm:\n\n')
    rng(88);
    cv_val = cvpartition(size(Xtrain, 1), 'HoldOut', 0.3);
    idx_val = cv_val.test;

    Xtrain1 = Xtrain(~idx_val, :);
    Xval  = Xtrain(idx_val, :);
    Ytrain1 = categorical(Ytrain(~idx_val, :));
    Yval = categorical(Ytrain(idx_val, :));
    
    % RBF / gaussian  --> onevsone (ovo) coding 
    rng(88); 
    classnames = {'1', '2'};

    min_error_rbf = 1;                  % Variable to record least errorcv
    val_error_rbf = zeros(4, 4);        % Validation error
    val_acc_rbf = zeros(4, 4);          % Validation Accuracy
    training_acc_rbf = zeros(4, 4);     % Training Accuracy 

    % G stands for Gamma, C stands for Box Constraint
    % These values represent the Grid Search range
    G = [0.1, 1, 10, 100];
    C = [0.1, 1, 10, 50];
 
    for i = 1:4
        
        for j = 1:4
        
            t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C(j), 'KernelScale', G(i));
            rbfMdl = fitcecoc(Xtrain1, Ytrain1, 'Learners', t, 'Coding', 'onevsone', ...
            'ClassNames', classnames);
            disp('Mdl done!')
        
            % Training Accuracy / Error
            prediction_rbf_train = predict(rbfMdl, Xtrain1);
            confusion_rbf_train = confusionmat(Ytrain1, categorical(prediction_rbf_train), 'Order', {'1', '2'});
    
            training_acc_rbf(i, j) = (confusion_rbf_train(1,1)+confusion_rbf_train(2,2)) / (sum(confusion_rbf_train, 'all'));
    
            disp('Training done!')
            disp(training_acc_rbf(i, j))
        
            % Validation Accuracy / Error 
            prediction_rbf_val = predict(rbfMdl, Xval);
            confusion_rbf_val = confusionmat(Yval, categorical(prediction_rbf_val), 'Order', {'1', '2'});
    
            val_acc_rbf(i, j) = (confusion_rbf_val(1,1)+confusion_rbf_val(2,2)) / (sum(confusion_rbf_val, 'all'));
        
            val_error_rbf(i, j) = 1 - val_acc_rbf(i, j); 

            disp('Validation done!')
            disp(val_acc_rbf(i, j))
        
            % Recording minimum validation error
            if val_error_rbf(i, j) < min_error_rbf
                min_error_rbf = val_error_rbf(i, j);
                best_i_rbf = G(i);
                best_j_rbf = C(j);
            end

        end
        disp(i)
        
    end

    writematrix(val_acc_rbf, 'val_acc_rbf.csv')
    writematrix(training_acc_rbf, 'training_acc_rbf.csv')
    fprintf('End of RBF svm.\n\n')

    %% SVM with Kernel Trick
    % Polynomial --> onevsone (ovo) coding
    fprintf('Start of Polynomial svm:\n\n')
    rng(88);
    classnames = {'1', '2'};

    min_error_poly = 1;                    % Variable to record least error
    val_error_poly = zeros(3, 4);          % Validation error
    val_acc_poly = zeros(3, 4);            % Validation Accuracy
    training_acc_poly = zeros(3, 4);       % Training Accuracy 

    i_index = 1;

    % C stands for Box Constraint
    % These values represent the Grid Search range
    C = [0.1, 1, 10, 50];

    % i refers to the polynomial order: [2,3,4]
    for i = 2:1:4
        
        for j = 1:4
            t = templateSVM('KernelFunction', 'polynomial', 'BoxConstraint', C(j), 'PolynomialOrder', i);
            polyMdl = fitcecoc(Xtrain1, Ytrain1, 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
            disp('Mdl done!')
        
            % Training Accuracy / Error
            prediction_poly_train = predict(polyMdl, Xtrain1);
            confusion_poly_train = confusionmat(Ytrain1, categorical(prediction_poly_train), 'Order', {'1', '2'});
        
            training_acc_poly(i_index, j) = (confusion_poly_train(1,1)+confusion_poly_train(2,2)) / (sum(confusion_poly_train, 'all'));   
    
            disp('Training done!')
            disp(training_acc_poly(i_index, j))
    
            % Validation Accuracy / Error 
            prediction_poly_val = predict(polyMdl, Xval);
            confusion_poly_val = confusionmat(Yval, categorical(prediction_poly_val), 'Order', {'1', '2'});
    
            val_acc_poly(i_index, j) = (confusion_poly_val(1,1)+confusion_poly_val(2,2)) / (sum(confusion_poly_val, 'all'));
        
            val_error_poly(i_index, j) = 1 - val_acc_poly(i_index, j);

            disp('Validation done!')
            disp(val_acc_poly(i_index, j))
        
            % Recording minimum validation error
            if val_error_poly(i_index, j) < min_error_poly
                min_error_poly = val_error_poly(i_index, j);
                best_i_poly = i;
                best_j_poly = C(j);
            end
                
        end
        disp(i)
        i_index = i_index + 1;
    
    end

    writematrix(val_acc_poly, 'val_acc_poly.csv')
    writematrix(training_acc_poly, 'training_acc_poly.csv')
    %save('SVM_GridSearch', 'training_acc_lin', 'val_acc_lin', 'C', ...
    %   'val_acc_rbf', 'training_acc_rbf', 'val_acc_poly', 'training_acc_poly')
    fprintf('End of Polynomial svm.\n\n')

    
    %% ************************************************************************
    %                       RBF SVM - GRID SEARCH
    % ************************************************************************

    % This script performs another Grid Search on specifically RBF SVM with KernelScale of 1. 
    % This RBF SVM is chosen as it produced the best accuracy scores in our first 
    % Grid Search process. To achieve an even greater score, we attempt to search for 
    % a best Box Constraint with the given model configurations. 
    
    % SVM RBF - Smaller Grid Search range for Box Constraints with CROSS VALIDATION
    % RBF --> onevsone (ovo) coding
    rng(88);
    classnames = {'1', '2'};

    C = [50, 60, 70, 75, 80, 85, 90, 100];
    k_folds = 5;
    Kernelscale = 1;

    cv_val = cvpartition(size(Ytrain, 1), 'KFold', k_folds);

    trainAvgScores = [];
    trainStdScores = [];
    valAvgScores = [];
    valStdScores = [];

    for j = 1:8
        trainScores = [];
        valScores = [];
    
        for iter = 1:k_folds
            % Model training
            t = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', C(j), 'KernelScale', Kernelscale);
            rbfMdl = fitcecoc(Xtrain(cv_val.training(iter), :), Ytrain(cv_val.training(iter), :), 'Learners', t, 'Coding', 'onevsone', 'ClassNames', classnames);
            disp('Mdl done!')
        
            % Get Train/Validation Scores (Accuracy)
            pred = str2double(predict(rbfMdl, Xtrain));
            trainScore = sum(pred(cv_val.training(iter))==Ytrain(cv_val.training(iter)))/length(pred(cv_val.training(iter)));
            valScore = sum(pred(cv_val.test(iter))==Ytrain(cv_val.test(iter))) / length(pred(cv_val.test(iter)));
        
            % Append Single Iteration Scores
            trainScores = [trainScores trainScore];
            valScores = [valScores valScore];
            disp('Prediction done!')
        end
        disp(j)
    
        % Append Average/Standard Deviation of Cross Validation Scores
        trainAvgScores = [trainAvgScores mean(trainScores)];
        trainStdScores = [trainStdScores std(trainScores)];
        valAvgScores = [valAvgScores mean(valScores)];
        valStdScores = [valStdScores std(valScores)];
    end

    % writematrix(valAvgScores, 'val_acc.csv')
    % writematrix(trainAvgScores, 'training_acc.csv')
    % save('RBF_Polynomial_GridSearch_#2', 'trainAvgScores', 'trainStdScores', 'valAvgScores', 'valStdScores', 'C')

    % Visualize Results
    figure(1);
    patch([C fliplr(C)], [trainAvgScores+trainStdScores, fliplr(trainAvgScores-trainStdScores)], [205/255 92/255 92/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    hold on;
    patch([C fliplr(C)], [valAvgScores+valStdScores, fliplr(valAvgScores-valStdScores)], [100/255 149/255 237/255], 'edgecolor', 'none', 'FaceAlpha', 0.2);
    line(C, trainAvgScores, 'color', [205/255 92/255 92/255], 'marker', '*', 'lineStyle', '-.');
    line(C, valAvgScores, 'color', [100/255 149/255 237/255], 'marker', '*', 'lineStyle', '-.');
    title('RBF SVM - Grid Search Results (Box Constraint)');
    legend('Train Accuracy Error', 'Validation Accuracy Error', 'Train Accuracy Estimate', 'Validation Accuraacy Estimate', 'Location', 'northwest');
    xlabel('Box Constraint');
    ylabel('Accuracy Score');
    textCell1 = arrayfun(@(x,y) sprintf('(%3.1f, %3.3f)',x,y),C,trainAvgScores,'un',0);
    for ii = 1:numel(C)
        text(C(ii)+.01, trainAvgScores(ii)+.002, textCell1{ii}, 'FontSize',8) 
    end
    textCell2 = arrayfun(@(x,y) sprintf('(%3.1f, %3.3f)',x,y),C,valAvgScores,'un',0);
    for ii = 1:numel(C)
        text(C(ii)+.01, valAvgScores(ii)+.002, textCell2{ii}, 'FontSize',8) 
    end

end