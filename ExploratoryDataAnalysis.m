% ************************************************************************
%                         Exploratory Data Analysis
% ************************************************************************

% This script performs an initial data analysis on the pre-processed dataset.
% Number of attributes: 9 (all are continuous)
% Number of classes: 2 (1 => negative, 2 => positive)

%% function for data analysis

function ExploratoryDataAnalysis (rawdata, smotedata)
    % Data shape
    [rawrows, rawcolumns] = size(rawdata);
    [smoterows, smotecolumns] = size(smotedata);
    fprintf("The original data contains %d observations with %d columns\n\n", rawrows, rawcolumns);
    fprintf("The smote data contains %d observations with %d columns\n\n", smoterows, smotecolumns);
    
    % Display first 5 rows of smote data
    disp("Display the first 5 rows of original data");
    head(smotedata, 5)
    
    % Missing values
    fprintf("There are no missing values in the dataset\n\n");
    
    % Summarize the data
    disp("Summary of raw data table:");
    summary(rawdata);
    disp("Summary of smote data table:");
    summary(smotedata);
    
    % Distribution of classes before and after smote
    figure('Name', "Distribution of Class", 'pos',[10 10 800 600])
    subplot(1,2,1)
    histogram(categorical(table2array(rawdata(:,end)), [1 2],{'negative' 'positive'}))
    title("Class distribution before smote")
    subplot(1,2,2)
    histogram(categorical(table2array(smotedata(:,end)), [1 2],{'negative' 'positive'}))
    title("Class distribution after smote")
    
    
    % Correlation between feature of data before smote
    corrMatrix = corr(table2array(rawdata(:, 1:end)),'type','Pearson');
    figure('Name', "Correlation Between Features Before Smote", 'pos',[100 100 800 800])
    imagesc(corrMatrix)
    colorbar; % enable colorbar
    colormap('parula'); % set the colorscheme
    
    n = length(rawdata.Properties.VariableNames);

    set(gca, 'XTick', 1:n); % center x-axis ticks on bins
    set(gca, 'YTick', 1:n); % center y-axis ticks on bins
    set(gca, 'XTickLabel', rawdata.Properties.VariableNames); % set x-axis labels
    set(gca, 'YTickLabel', rawdata.Properties.VariableNames); % set y-axis labels
    xtickangle(45) % set angle of xticks to 45
    % set title
    title('Pearson Correlation Plot of Attributes Before Smote', 'FontSize', 14);
    
    
     % Correlation between features of data after smote
    corrMatrix = corr(table2array(smotedata(:, 1:end)),'type','Pearson');
    figure('Name', "Correlation Between Features After Smote", 'pos',[100 100 800 800])
    imagesc(corrMatrix)
    colorbar; % enable colorbar
    colormap('parula'); % set the colorscheme
    
    n = length(smotedata.Properties.VariableNames);

    set(gca, 'XTick', 1:n); % center x-axis ticks on bins
    set(gca, 'YTick', 1:n); % center y-axis ticks on bins
    set(gca, 'XTickLabel', smotedata.Properties.VariableNames); % set x-axis labels
    set(gca, 'YTickLabel', smotedata.Properties.VariableNames); % set y-axis labels
    xtickangle(45) % set angle of xticks to 45
    % set title
    title('Pearson Correlation Plot of Attributes After Smote', 'FontSize', 14);
    
end

