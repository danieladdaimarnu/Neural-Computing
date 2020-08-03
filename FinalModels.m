% ************************************************************************
%                        FINAL MODEL - TEST COMPARISON
% ************************************************************************

% This script performs a final comparison on optimised models.
% ROC curves and confusion matrices are produced to compare each performance. 


function FinalModels(trainData, testData)
    
    % Data Processing
    Xtrain = table2array(normalize(trainData(:, 1:end-1)));
    Ytrain = table2array(trainData(:, end));
    Xtest = table2array(normalize(testData(:, 1:end-1)));
    Ytest = table2array(testData(:, end));
    yOHEtrain = dummyvar(Ytrain);
    yOHEtest = dummyvar(Ytest);
    
    % Parameters
    % MLP
    netSize = 34;
    trainFcn = 'trainscg';
    % SVM
    Kernelscale = 1;
    boxConstraint = 80;
    classnames = {'1', '2'};
    
    % Define Model
      % MLP
    net = feedforwardnet(netSize, trainFcn);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'softmax';
    net.performFcn = 'crossentropy';
    net.plotFcns = {'plotperform'};
      % SVM
    SVM_Template = templateSVM('KernelFunction', 'rbf', 'BoxConstraint', boxConstraint, 'KernelScale', Kernelscale);
    
    % Train Model
                % MLP
    [net tr] = train(net, Xtrain', yOHEtrain');
    
                % SVM
    svm = fitcecoc(Xtrain, Ytrain, 'Learners', SVM_Template, 'FitPosterior', true, 'Coding', 'onevsone',...
        'ClassNames', classnames);
  
    %% Confusion Matrices and Accuracy Scores
        % MLP
    % Get Test Scores (Accuracy)
    MLP_pred_test = net(Xtest');
    target_test_ind = vec2ind(yOHEtest');
    MLP_pred_test_ind = vec2ind(MLP_pred_test);
    MLP_testScore = sum(target_test_ind == MLP_pred_test_ind)/numel(target_test_ind);
    

        % SVM
    % Get Test Scores (Accuracy)
    [SVM_pred,~,~,svmPost] = predict(svm,Xtest);
    SVM_pred_test = str2double(SVM_pred);
    SVM_testScore = sum(SVM_pred_test==Ytest)/length(SVM_pred_test);
    
    fprintf('Test Score -----------\n')
    fprintf('SVM    : %.3f\n', SVM_testScore)
    fprintf('MLP    : %.3f\n\n', MLP_testScore)
    
    %% Visualise Results
    
    % Confusion Matrix
    figure(1);
    plotconfusion(categorical(Ytest),categorical(SVM_pred_test),'SVM',yOHEtest', MLP_pred_test,'MLP')
    
    % ROC Curve SVM vs MLP
    [Xsvm,Ysvm,Tsvm,AUCsvm,OPTsvm] = perfcurve(Ytest,svmPost(:,1),1); % SVM
    [Xmlp,Ymlp,Tmlp,AUCmlp,OPTmlp] = perfcurve(target_test_ind,MLP_pred_test(1,:),1); % MLP
    
    fprintf('AUC Prediction Performance -----------\n')
    fprintf('SVM    : %.3f\n', AUCsvm)
    fprintf('MLP    : %.3f\n\n', AUCmlp)
    
    % Plot the ROC curves on the same graph.
    figure(2);
    plot(Xsvm,Ysvm, 'b')
    hold on
    plot(Xmlp,Ymlp, 'r')
    % plot Optimal operating point of the ROC curve
    plot(OPTsvm(1), OPTsvm(2),'bo')
    plot(OPTmlp(1), OPTmlp(2),'ro')
    grid on
    % write AUC on plot
    text(0.3,0.35,strcat('SVM AUC = ',num2str(AUCsvm)),'EdgeColor','b')
    text(0.3,0.30,strcat('MLP AUC = ',num2str(AUCmlp)),'EdgeColor','r')
    legend('SVM', 'MLP', 'SVM OPTROCPT',...
         'MPL OPTROCPT', 'Location','Best')
    
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for SVM and MLP')
    hold off
    

    %% Saving Variables
    %save('results_final_models.mat', 'net', 'svm', 'MLP_confusion_test', 'SVM_confusion_test', 'MLP_testScore', 'SVM_testScore')
end