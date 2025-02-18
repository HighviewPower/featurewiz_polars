# polars_verifier_mrmr_v6.py (Minor Clarity Tweaks in MRMR)
import numpy as np
import pandas as pd
import polars as pl
np.random.seed(42)
import polars.selectors as cs
import pyarrow
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
# Needed imports
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # Or LightGBM, XGBoost, etc.
# Import the Polars CategoricalFeatureEncoderV2 and PolarsMRMRSelectorV3 classes
from polars_categorical_encoder import Polars_CategoricalEncoder # Now using V2 of Encoder
from polars_datetime_transformer import Polars_DateTimeTransformer # Import new transformer
from polars_sulov_mrmr import Sulov_MRMR
from polars_other_transformers import YTransformer, Polars_MissingTransformer, Polars_ColumnEncoder
import time
import pdb
#############################################################################
class Featurewiz_MRMR(BaseEstimator, TransformerMixin): # Class name 
    def __init__(self, 
            model_type='classification', encoding_type='target', 
            imputation_strategy='mean', corr_threshold = 0.7,
            verbose = 0):
        self.model_type = model_type.lower()
        self.encoding_type = encoding_type.lower()
        self.imputation_strategy = imputation_strategy.lower()
        self.corr_threshold = corr_threshold
        self.verbose = verbose
        self.preprocessing_pipeline = None
        self.featurewiz_pipeline = None
        self.feature_selection = None
        self.selected_features = []
        # MRMR is different for regression and classification
        if self.model_type == 'regression':
            ### This is for Regression where no YTransformer is needed ##
            preprocessing_pipeline = Pipeline([
                    ('datetime_transformer', Polars_DateTimeTransformer(datetime_features="auto")), # Specify your datetime columns
                    ('cat_transformer', Polars_CategoricalEncoder(encoding_type=self.encoding_type, categorical_features="auto", handle_unknown='value', unknown_value=0.0)),
                    ('nan_transformer', Polars_MissingTransformer(strategy=self.imputation_strategy)),
                ])
        else:
            #### This is for Classification where YTransformer is needed ####
            #### You need YTransformer in the X_pipeline becasue featurewiz uses XGBoost which needs a transformed Y. Otherwise error!
            preprocessing_pipeline = Pipeline([
                    ('datetime_transformer', Polars_DateTimeTransformer(datetime_features="auto")), # Specify your datetime columns
                    ('cat_transformer', Polars_CategoricalEncoder(encoding_type=self.encoding_type, categorical_features="auto", handle_unknown='value', unknown_value=0.0)),
                    ('nan_transformer', Polars_MissingTransformer(strategy=self.imputation_strategy)),
                    ('ytransformer', YTransformer()),
                ])

        featurewiz_pipeline = Pipeline([
                    ('featurewiz', Sulov_MRMR(corr_threshold=self.corr_threshold, model_type=self.model_type, verbose=self.verbose)),
                ])

        feature_selection = Pipeline([
                ('PreProcessing_pipeline', preprocessing_pipeline),
                ('Featurewiz_pipeline', featurewiz_pipeline)
            ])

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder

    def _check_pandas(self, X):
        if isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
        else:
            return X

    def fit(self, X, y):
        start_time = time.time()
        X = self._check_pandas(X)
        y = self._check_pandas(y)

        #### Now train the model using the feature union pipeline
        self.feature_selection.fit(X, y)
        self.y_encoder.fit(y)

        ### If successful save all the pipelines in following variables to use later in transform and predict
        self.preprocessing_pipeline = self.feature_selection[-2]
        self.featurewiz_pipeline = self.feature_selection[-1]
        ### since this is a pipeline within a pipeline, you have to use nested lists to get the features!
        self.selected_features = self.feature_selection[-1][-1].get_feature_names_out()
        print('\nPolars Featurewiz MRMR completed. Time taken  = %0.1f seconds' %(time.time()-start_time))
        print('    Use "selected_features" attribute to retrieve list of selected features from featurewiz pipeline')
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Transforms the data by selecting the top MRMR features in Polars DataFrame.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y: optional since it may not be available for test data

        Returns:
            Polars DataFrame: Transformed DataFrame with selected features, shape (n_samples, n_selected_features)
            Polars Series: Transformed Series if y is given
        """
        check_is_fitted(self, 'fitted_')
        X = self._check_pandas(X)
        if y is None:
            return self.feature_selection.transform(X)
        else:
            Xt = self.feature_selection.transform(X)
            yt = self.y_encoder.transform(y)
            return Xt, yt

    def fit_transform(self, X, y):
        """
        Fits and Transforms the data by selecting the top MRMR features in Polars DataFrame.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y: is not optional since it is required for Recursive XGBoost feature selection

        Returns:
            Polars DataFrame: Transformed DataFrame with selected features, shape (n_samples, n_selected_features)
            Polars Series: Transformed Series if y is given
        """
        self.fit(X, y)
        Xt = self.transform(X)
        yt = self.y_encoder.transform(y)
        return Xt, yt

##############################################################################
class Featurewiz_MRMR_Model(BaseEstimator, TransformerMixin): # Class name 
    def __init__(self, model=None, 
            model_type='classification', encoding_type='target', 
            imputation_strategy='mean', corr_threshold = 0.7,
            verbose = 0):
        self.model = model
        self.model_type = model_type.lower()
        self.encoding_type = encoding_type.lower()
        self.imputation_strategy = imputation_strategy.lower()
        self.corr_threshold = corr_threshold
        self.verbose = verbose
        self.preprocessing_pipeline = None
        self.featurewiz_pipeline = None
        self.feature_selection = None
        self.selected_features = []
        # MRMR is different for regression and classification
        feature_selection = Featurewiz_MRMR(model_type=self.model_type, 
            encoding_type=self.encoding_type, 
            imputation_strategy=self.imputation_strategy, 
            corr_threshold =self.corr_threshold,
            verbose=self.verbose)

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder

    def _check_pandas(self, X):
        if isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
        else:
            return X

    def fit(self, X, y):
        start_time = time.time()
        X = self._check_pandas(X)
        y = self._check_pandas(y)

        #### Now train the model using the feature union pipeline
        self.feature_selection.fit(X, y)
        self.y_encoder.fit(y)
        
        ### since this is a pipeline within a pipeline, you have to use nested lists to get the features!
        self.selected_features = self.feature_selection.selected_features
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Transforms the data by selecting the top MRMR features in Polars DataFrame.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y: optional since it may not be available for test data

        Returns:
            Polars DataFrame: Transformed DataFrame with selected features, shape (n_samples, n_selected_features)
        """
        check_is_fitted(self, 'fitted_')
        X = self._check_pandas(X)
        if y is None:
            return self.feature_selection.transform(X)
        else:
            y = self._check_pandas(y)
            Xt = self.feature_selection.transform(X)
            yt = self.y_encoder.transform(y)
            self.model.fit(Xt, yt)
            return Xt, yt

    def fit_transform(self, X, y):
        X = self._check_pandas(X)
        y = self._check_pandas(y)
        self.fit(X, y)
        Xt = self.transform(X)
        yt = self.y_encoder.transform(y)
        self.model.fit(Xt, yt)
        return Xt, yt

    def fit_predict(self, X, y):
        X = self._check_pandas(X)
        y = self._check_pandas(y)
        self.fit(X, y)
        if not self.model is None:
            Xt = self.transform(X)
            yt = self.y_encoder.transform(y)
            self.model.fit(Xt, yt)
            return self.model.predict(Xt)
        else:
            raise ValueError("Inappropriate value of None for model argument in pipeline. Please correct and try again.")

    def predict(self, X, y=None) :
        """
        Predicts on the data by selecting the top MRMR features in Polars DataFrame.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y: optional since it may not be available for test data

        Returns:
            Polars DataFrame: Transformed DataFrame with selected features, shape (n_samples, n_selected_features)
        """
        check_is_fitted(self, 'fitted_')
        X = self._check_pandas(X)
        Xt = self.transform(X)
        return self.model.predict(Xt)
##############################################################################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import cycle
def print_regression_model_stats(actuals, predicted, verbose=0):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    if isinstance(actuals,pd.Series) or isinstance(actuals,pd.DataFrame):
        actuals = actuals.values
    if isinstance(predicted,pd.Series) or isinstance(predicted,pd.DataFrame):
        predicted = predicted.values
    if len(actuals) != len(predicted):
        if verbose:
            print('Error: Number of rows in actuals and predicted dont match. Continuing...')
        return np.inf
    try:
        ### This is for Multi_Label Problems ###
        assert actuals.shape[1]
        multi_label = True
    except:
        multi_label = False
    if multi_label:
        for i in range(actuals.shape[1]):
            actuals_x = actuals[:,i]
            try:
                predicted_x = predicted[:,i]
            except:
                predicted_x = predicted[:]
            if verbose:
                print('for target %s:' %i)
            each_rmse = print_regression_metrics(actuals_x, predicted_x, verbose)
        final_rmse = np.mean(each_rmse)
    else:
        final_rmse = print_regression_metrics(actuals, predicted, verbose)
    return final_rmse
################################################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def MAPE(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

def print_regression_metrics(y_true, y_preds, verbose=0):
    try:
        each_rmse = np.sqrt(mean_squared_error(y_true, y_preds))
        if verbose:
            print('    RMSE = %0.3f' %each_rmse)
            print('    Norm RMSE = %0.0f%%' %(100*np.sqrt(mean_squared_error(y_true, y_preds))/np.std(y_true)))
            print('    MAE = %0.3f'  %mean_absolute_error(y_true, y_preds))
        if len(y_true[(y_true==0)]) > 0:
            if verbose:
                print('    WAPE = %0.0f%%, Bias = %0.1f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                            100*np.sum(y_true-y_preds)/np.sum(y_true)))
                print('    No MAPE available since zeroes in actuals')
        else:
            if verbose:
                print('    WAPE = %0.0f%%, Bias = %0.1f%%' %(100*np.sum(np.abs(y_true-y_preds))/np.sum(y_true), 
                            100*np.sum(y_true-y_preds)/np.sum(y_true)))
                print('    MAPE = %0.0f%%' %(100*MAPE(y_true, y_preds)))
        print('    R-Squared = %0.0f%%' %(100*r2_score(y_true, y_preds)))
        if not verbose:
            plot_regression(y_true, y_preds, chart='scatter')
        return each_rmse
    except Exception as e:
        print('Could not print regression metrics due to %s.' %e)
        return np.inf
################################################################################
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev
################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    To avoid infinity due to division by zero, we select max(0.01, abs(actuals)) to show MAPE.
    """
    ### Wherever there is zero, replace it with 0.001 so it doesn't result in division by zero
    perc_err = (100*(np.where(y==0,0.001,y) - y_hat))/np.where(y==0,0.001,y)
    return np.mean(abs(perc_err))
    
def plot_regression(actuals, predicted, chart='scatter'):
    """
    This function plots the actuals vs. predicted as a line plot.
    You can change the chart type to "scatter' to get a scatter plot.
    """
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    plt.figure(figsize=figsize)
    if not isinstance(actuals, np.ndarray):
        actuals = actuals.values
    dfplot = pd.DataFrame([actuals,predicted]).T
    dfplot.columns = ['Actual','Forecast']
    dfplot = dfplot.sort_index()
    lineStart = actuals.min()
    lineEnd = actuals.max()
    if chart == 'line':
        plt.plot(dfplot)
    else:
        plt.scatter(actuals, predicted, color = next(colors), alpha=0.5,label='Predictions')
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
        plt.xlim(lineStart, lineEnd)
        plt.ylim(lineStart, lineEnd)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Model: Predicted vs Actuals', fontsize=12)
    plt.show();
###########################################################################
from sklearn.metrics import roc_auc_score
import copy
from sklearn.metrics import balanced_accuracy_score, classification_report
import pdb
def print_sulo_accuracy(y_test, y_preds, y_probas='', verbose=0):
    """
    A wrapper function for print_classification_metrics,  meant for compatibility with older featurewiz versions.
    Usage:
    print_sulo_accuracy(y_test, y_preds, y_probas, verbose-0)
    """
    return print_classification_metrics(y_test, y_preds, y_probas, verbose)

def print_classification_metrics(y_test, y_preds, y_probas='', verbose=0):
    """
    Calculate and print classification metrics for single-label, multi-label, and multi-class problems.

    This function computes and displays various metrics such as balanced accuracy score and ROC AUC score 
    based on the given test labels, predicted labels, and predicted probabilities. It handles different 
    scenarios including single-label, multi-label, multi-class, and their combinations. Additionally, it 
    provides detailed classification reports if verbose output is requested.

    Parameters:
    y_test (array-like): True labels. Should be 1D for single-label and 2D for multi-label problems.
    y_preds (array-like): Predicted labels. Should match the dimensionality of y_test.
    y_probas (array-like, optional): Predicted probabilities. Default is an empty string, indicating 
                                     no probabilities are provided. Should be 2D with probabilities for 
                                     each class.
    verbose (int, optional): Verbose level. If set to 1, it prints out detailed classification reports. 
                             Default is 0, which prints only summary metrics.

    Returns:
    float: Final average balanced accuracy score across all labels/classes. Returns 0.0 if an exception occurs.

    Raises:
    Exception: If an error occurs during the calculation or printing of metrics.

    Note:
    The function is designed to handle various edge cases and different formats of predicted probabilities, 
    such as those produced by different classifiers or methods like Label Propagation.

    Examples:
    # For single-label binary classification
    print_classification_metrics(y_test, y_preds)

    # For multi-label classification with verbose output
    print_classification_metrics(y_test, y_preds, verbose=1)

    # For single-label classification with predicted probabilities
    print_classification_metrics(y_test, y_preds, y_probas)
    """    
    try:
        bal_scores = []
        ####### Once you have detected what problem it is, now print its scores #####
        if y_test.ndim <= 1: 
            ### This is a single label problem # we need to test for multiclass ##
            bal_score = balanced_accuracy_score(y_test,y_preds)
            print('Bal accu %0.0f%%' %(100*bal_score))
            if not isinstance(y_probas, str):
                if y_probas.ndim <= 1:
                    print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                else:
                    if y_probas.shape[1] == 2:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                    else:
                        print('Multi-class ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
            bal_scores.append(bal_score)
            if verbose:
                print(classification_report(y_test,y_preds))
        elif y_test.ndim >= 2:
            if y_test.shape[1] == 1:
                bal_score = balanced_accuracy_score(y_test,y_preds)
                bal_scores.append(bal_score)
                print('Bal accu %0.0f%%' %(100*bal_score))
                if not isinstance(y_probas, str):
                    if y_probas.shape[1] > 2:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas, multi_class="ovr"))
                    else:
                        print('ROC AUC = %0.2f' %roc_auc_score(y_test, y_probas[:,1]))
                if verbose:
                    print(classification_report(y_test,y_preds))
            else:
                if isinstance(y_probas, str):
                    ### This is for multi-label problems without probas ####
                    for each_i in range(y_test.shape[1]):
                        bal_score = balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])
                        bal_scores.append(bal_score)
                        print('    Bal accu %0.0f%%' %(100*bal_score))
                        if verbose:
                            print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
                else:
                    ##### This is only for multi_label_multi_class problems
                    num_targets = y_test.shape[1]
                    for each_i in range(num_targets):
                        print('    Bal accu %0.0f%%' %(100*balanced_accuracy_score(y_test.values[:,each_i],y_preds[:,each_i])))
                        if len(np.unique(y_test.values[:,each_i])) > 2:
                            ### This nan problem happens due to Label Propagation but can be fixed as follows ##
                            mat = y_probas[each_i]
                            if np.any(np.isnan(mat)):
                                mat = pd.DataFrame(mat).fillna(method='ffill').values
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                            else:
                                bal_score = roc_auc_score(y_test.values[:,each_i],mat,multi_class="ovr")
                        else:
                            if isinstance(y_probas, dict):
                                if y_probas[each_i].ndim <= 1:
                                    ## This is caused by Label Propagation hence you must probas like this ##
                                    mat = y_probas[each_i]
                                    if np.any(np.isnan(mat)):
                                        mat = pd.DataFrame(mat).fillna(method='ffill').values
                                    bal_score = roc_auc_score(y_test.values[:,each_i],mat)
                                else:
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i][:,1])
                            else:
                                if y_probas.shape[1] == num_targets:
                                    ### This means Label Propagation was used which creates probas like this ##
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[:,each_i])
                                else:
                                    ### This means regular sklearn classifiers which predict multi dim probas #
                                    bal_score = roc_auc_score(y_test.values[:,each_i],y_probas[each_i])
                        print('Target number %s: ROC AUC score %0.0f%%' %(each_i+1,100*bal_score))
                        bal_scores.append(bal_score)
                        if verbose:
                            print(classification_report(y_test.values[:,each_i],y_preds[:,each_i]))
        final_score = np.mean(bal_scores)
        if verbose:
            print("final average balanced accuracy score = %0.2f" %final_score)
        return final_score
    except Exception as e:
        print('Could not print classification metrics due to %s.' %e)
        return 0.0
######################################################################################################
def split_polars_dataframe(df, target_column_name, test_size=0.2):
	# 1. Generate a random sample index for the training set
	train_fraction = 1-test_size  # 80% for training, 20% for testing
	n_rows = df.height
	train_size = int(n_rows * train_fraction)

	print("Splitting Polars dataframe using Polars native functions...")
	train_df = df.slice(0,train_size)
	test_df = df.slice(train_size)

	# 3. Separate features (X) and target (y) for both sets
	X_train_pl = train_df.drop([target_column_name]) # Drop target and row_nr for features
	y_train_pl = train_df[target_column_name]
	X_test_pl = test_df.drop([target_column_name])   # Drop target and row_nr for features
	y_test_pl = test_df[target_column_name]

	print("\nX_train_pl (Polars DataFrame - Features - Training Set):")
	print(X_train_pl.shape)
	print("\ny_train_pl (Polars Series - Target - Training Set):")
	print(y_train_pl.shape)
	print("\nX_test_pl (Polars DataFrame - Features - Testing Set):")
	print(X_test_pl.shape)
	print("\ny_test_pl (Polars Series - Target - Testing Set):")
	print(y_test_pl.shape)
	return (X_train_pl, X_test_pl, y_train_pl, y_test_pl)
######################################################################################################
