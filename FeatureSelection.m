% Rank features for classification using decision trees
function FeatureSelection(data, targetCol)
    % Train a classification tree using the entire data set. To grow
    % unbiased trees, specify usage of the curvature test for splitting predictors. 
    
    rng(5);
    Mdl = fitctree(data, targetCol,'PredictorSelection','curvature','Surrogate','on');
    % Estimate predictor importance values by summing changes in the risk
    % due to splits on every predictor and dividing the sum by the number of branch nodes.
    imp = predictorImportance(Mdl);
    fprintf('Feature importance of attributes: \n');
    T = categorical(Mdl.PredictorNames);
    A = categorical(imp);
    disp([T',A']);
    % visualize feature importance
    figure('Name', "Feature Importance Using Decision Tree", 'pos',[50 50 800 600]);
    bar(T,imp);
    title('Feature importance using decision tree');
    ylabel('Estimates');
    xlabel('Predictors');
    h = gca;
    h.XTickLabelRotation = 45;
    h.TickLabelInterpreter = 'none';
    disp("In the case of decision tree, 'kurt_ip' is the most important predictor.")
end