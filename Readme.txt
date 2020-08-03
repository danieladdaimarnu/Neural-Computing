
################################################################
			     README
################################################################


##############
# Introduction
##############

This Readme file contains instructions and remarks concerning the coursework deliverables developed by Daniel Addai-Marnu. Please take a moment to read it first before moving on.


################
# Matlab Version
################

All the work has been developed using Matlab R2019b. It should be compatible with future versions, but I did NOT test that.


########################
# Deliverables Structure
########################

In the present folder, you will find :
	- Two folders : Scripts and Data
	- main.m : Central/main script to call and run all scripts
	- One .txt file : Readme.txt
	- One .pdf file : Coursework Report - NCcoursework.pdf

In the Scripts folder, you will find :
	- Scripts : each one is dedicated to an experiment or task. They can all be called and run through main.m script.

#######################
# Experiments & Scripts
#######################

The experiment scripts are presented as follows :

	--> ExploratoryDataAnalysis.m : Returns some statistics about the data, mainly Correlation Heatmap (NOT included within report) and Product Count plot (before and after smote). 

	--> FeatureSelection.m : Returns the feature importance of the independent variables in predicting the dependent variable using decision tree.

	--> DecisionBoundary.m : Returns the decision boundaries of the models.

	--> SVMHyperParameterTuning.m : Hyper-parameter tuning of the SVM model.

	--> MLPHyperParameterTuning.m : Hyper-parameter tuning of the MLP model.
		
		* cvLoss.m : This script defines the Objective Function that will be used in the bayesian Optimisation procedure for Hyper-Parameter tuning. It builds a Neural Network and evaluates its performance on a Holdout set.

	--> LearningCurve.m : Returns the visualisation of the Learning Curves of the optimised MLP and SVM models. 10-fold cross validation is used to output accuracy estimates for both training and validation along with their estimated errors as represented by the standard deviation.

	--> FinalModels.m : Performs a final comparison between optimised SVM and MLP models given the same unseen test data. 
	

	--> SVMSpaceTimeComplexity.m : Performs a complexity analysis (both space and time) on SVM, mainly in the form of plots for visualisation clarity in presentation. 

	--> SVMSpaceTimeComplexity.m : Performs a complexity analysis (both space and time) on MLP, mainly in the form of plots for visualisation clarity in presentation. 






