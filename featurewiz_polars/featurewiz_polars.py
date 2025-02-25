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
# Import the Polars classes now
from .polars_categorical_encoder import Polars_CategoricalEncoder # Now using V2 of Encoder
from .polars_datetime_transformer import Polars_DateTimeTransformer # Import new date-time transformer
from .polars_other_transformers import YTransformer, Polars_MissingTransformer, Polars_ColumnEncoder
from .polars_sulov_mrmr import Sulov_MRMR
import time
import matplotlib.pyplot as plt
import pdb
#############################################################################
class Featurewiz_MRMR(BaseEstimator, TransformerMixin): # Class name 
    def __init__(self, 
            model_type='classification', encoding_type='target', 
            imputation_strategy='mean', corr_threshold = 0.7,
            classic = False,
            verbose = 0):
        """
        Initializes the Featurewiz_MRMR class for feature engineering and selection.

        Args:
            model_type (str, optional):  The type of model to be built ('classification' or 'regression'). 
            Determines the appropriate preprocessing and feature selection strategies. Defaults to 'classification'.

            encoding_type (str, optional): The type of encoding to apply to categorical features ('woe', 'target', 'ordinal', 'onehot', etc.).  
            'woe' encoding is only available for classification model types. Defaults to 'target'.

            imputation_strategy (str, optional): The strategy for handling missing values ('mean', 'median', 'zeros'). 
            Determines how missing data will be filled in before feature selection. Defaults to 'mean'.

            corr_threshold (float, optional): The correlation threshold for removing highly correlated features. 
            Features with a correlation above this threshold will be considered for removal. Defaults to 0.7.

            classic (bool, optional): If true, it implements the original classic featurewiz approach with recursive_xgboost implemented in Polars.
            If False, implements the train-validation-recursive-xgboost version, which is slower and uses train_test_split schemes to stabilize features.

            verbose (int, optional): Controls the verbosity of the output during feature selection.  
            0 for minimal output, 1 for more detailed information and 2 for very detailed info. Defaults to 0.
        """
        self.model_type = model_type.lower()
        self.encoding_type = encoding_type.lower()
        self.imputation_strategy = imputation_strategy.lower()
        self.corr_threshold = corr_threshold
        self.verbose = verbose
        self.preprocessing_pipeline = None
        self.featurewiz_pipeline = None
        self.feature_selection = None
        self.selected_features = []
        self.classic = classic
        # MRMR is different for regression and classification
        if self.model_type == 'regression':
            
            ### This is for Regression where no YTransformer is needed ##
            preprocessing_pipeline = Pipeline([
                    ('datetime_transformer', Polars_DateTimeTransformer(datetime_features="auto")), # Specify your datetime columns
                    ('cat_transformer', Polars_CategoricalEncoder(model_type=self.model_type, encoding_type=self.encoding_type, categorical_features="auto", handle_unknown='value', unknown_value=0.0)),
                    ('nan_transformer', Polars_MissingTransformer(strategy=self.imputation_strategy)),
                ])
        else:
            #### This is for Classification where YTransformer is needed ####
            #### You need YTransformer in the X_pipeline becasue featurewiz uses XGBoost which needs a transformed Y. Otherwise error!
            preprocessing_pipeline = Pipeline([
                    ('datetime_transformer', Polars_DateTimeTransformer(datetime_features="auto")), # Specify your datetime columns
                    ('cat_transformer', Polars_CategoricalEncoder(model_type=self.model_type, encoding_type=self.encoding_type, categorical_features="auto", handle_unknown='value', unknown_value=0.0)),
                    ('nan_transformer', Polars_MissingTransformer(strategy=self.imputation_strategy)),
                    ('ytransformer', YTransformer()),
                ])

        featurewiz_pipeline = Pipeline([
                    ('featurewiz', Sulov_MRMR(corr_threshold=self.corr_threshold, model_type=self.model_type, classic=self.classic, verbose=self.verbose)),
                ])

        feature_selection = Pipeline([
                ('PreProcessing_pipeline', preprocessing_pipeline),
                ('Featurewiz_pipeline', featurewiz_pipeline)
            ])

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder

    def _check_pandas(self, XX):
        """
        Converts Pandas DataFrames/Series to Polars DataFrames.

        Args:
            XX (pd.DataFrame, pd.Series, or pl.DataFrame): The input data.

        Returns:
            pl.DataFrame: A Polars DataFrame. If the input was already a Polars DataFrame, it is returned unchanged.

        Notes:
            - This method checks if the input data (XX) is a Pandas DataFrame or Series.
            - If it is, it converts the data to a Polars DataFrame using `pl.from_pandas()`.
            - If the input is already a Polars DataFrame or is of a different type, it is returned without modification.
        """
        if isinstance(XX, pd.DataFrame) or isinstance(XX, pd.Series):
            return pl.from_pandas(XX)
        else:
            return XX

    def fit(self, X, y):
        """
        Fits the Featurewiz_MRMR class to the input data. This performs the core feature engineering and selection steps.

        Args:
            X (pd.DataFrame or pl.DataFrame): The input feature data.  Can be a Pandas or Polars DataFrame.
            y (pd.Series or pl.Series): The target variable. Can be a Pandas or Polars Series.

        Returns:
            self: Returns an instance of self after fitting.

        Raises:
            TypeError: If X or y are not Polars DataFrames/Series.

        Notes:
            - Internally, this method:
                1. Fits the feature selection pipeline to the data.
                2. Fits the y converter (encoder) to the target variable.
                3. Stores the trained preprocessing and featurewiz pipelines for later use in `transform` and `predict`.
                4. Extracts the names of the selected features from featurewiz.
        """
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
        print('\nFeaturewiz Polars MRMR completed. Time taken  = %0.1f seconds' %(time.time()-start_time))
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
    """
    Initializes the Featurewiz_MRMR_Model class for feature engineering, selection, and model training.

    Args:
        model (estimator object, optional): Any machine learning estimator can be sent in to be trained after feature selection.
        If None, a default estimator will be used (e.g., Random Forest). Defaults to None.

        model_type (str, optional): The type of model to be built ('classification' or 'regression').
        Determines the appropriate preprocessing and feature selection strategies. Defaults to 'classification'.

        encoding_type (str, optional): The type of encoding to apply to categorical features ('target', 'onehot', etc.).
        'woe' encoding is only available for classification model_types. Defaults to 'target'.

        imputation_strategy (str, optional): The strategy for handling missing values ('mean', 'median', 'zeros').
        Determines how missing data will be filled in before feature selection. Defaults to 'mean'.

        corr_threshold (float, optional): The correlation threshold for removing highly correlated features.
        Features with a correlation above this threshold will be targeted for removal. Defaults to 0.7.

        classic (bool, optional): If true, it implements the original classic featurewiz library using Polars. 
        If False, implements train-validation-split-recursive-xgboost version, which is slower and uses train_test_splits to stabilize features.

        verbose (int, optional): Controls the verbosity of the output during feature selection. 
        0 for minimal output, higher values for more detailed information. Defaults to 0.
    """    
    def __init__(self, model=None, 
        model_type='classification', encoding_type='target', 
        imputation_strategy='mean', corr_threshold = 0.7,
        classic=False,
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
        self.model_fitted_ = False
        self.classic = classic
        # MRMR is same for regression and classification
        feature_selection = Featurewiz_MRMR(model_type=self.model_type, 
            encoding_type=self.encoding_type, 
            imputation_strategy=self.imputation_strategy, 
            corr_threshold =self.corr_threshold,
            classic=self.classic,
            verbose=self.verbose)

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder

    def _check_pandas(self, XX):
        """
        Converts Pandas DataFrames/Series to Polars DataFrames.

        Args:
            XX (pd.DataFrame, pd.Series, or pl.DataFrame): The input data.

        Returns:
            pl.DataFrame: A Polars DataFrame. If the input was already a Polars DataFrame, it is returned unchanged.

        Notes:
            - This method checks if the input data (XX) is a Pandas DataFrame or Series.
            - If it is, it converts the data to a Polars DataFrame using `pl.from_pandas()`.
            - If the input is already a Polars DataFrame or is of a different type, it is returned without modification.
        """
        if isinstance(XX, pd.DataFrame) or isinstance(XX, pd.Series):
            return pl.from_pandas(XX)
        else:
            return XX

    def fit(self, X, y):
        """
        Fits the Featurewiz_MRMR_Model to the input data and trains the specified model.

        Args:
            X (pd.DataFrame or pl.DataFrame): The input feature data. Can be a Pandas or Polars DataFrame.
            y (pd.Series or pl.Series): The target variable. Can be a Pandas or Polars Series.

        Returns:
            self: Returns an instance of self after fitting.

        Raises:
            TypeError: If X or y are not Polars DataFrames/Series.

        Notes:
            - This method performs the following steps:
                1. Converts X and y to Polars DataFrames if they are Pandas DataFrames.
                2. Fits the feature selection pipeline to the data using `self.feature_selection.fit(X, y)`.
                3. Fits the target encoder to the target variable using `self.y_encoder.fit(y)`.
                4. If a model was not provided during initialization, a default RandomForestRegressor (for regression) or RandomForestClassifier (for classification) is created.
                5. Trains the model using the selected features: `self.model.fit(X[self.selected_features], y)`.
                6. Sets `self.model_fitted_` to True to indicate that the model has been trained.
        """
        start_time = time.time()
        X = self._check_pandas(X)
        y = self._check_pandas(y)

        #### Now train the model using the feature union pipeline
        self.feature_selection.fit(X, y)
        self.y_encoder.fit(y)
        if self.model_type == 'regression':
            ### The model is not fitted yet so self.model_fitted_ is still False
            if self.model is None:
                self.model = RandomForestRegressor(n_estimators=100, random_state=99)
        else:
            ### The model is not fitted yet so self.model_fitted_ is still False
            if self.model is None:
                self.model = RandomForestClassifier(n_estimators=100, random_state=99)
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
            self.model.fit(Xt[self.selected_features], yt)
            ### The model is fitted now so self.model_fitted_ is set to True
            self.model_fitted_ = True
            return Xt, yt

    def fit_transform(self, X, y):
        """
        Fits the Featurewiz_MRMR_Model to the input data, transforms the data, and fits the model to the transformed data. This is a combined operation for convenience.

        Args:
            X (pd.DataFrame or pl.DataFrame): The input feature data. Can be a Pandas or Polars DataFrame.
            y (pd.Series or pl.Series): The target variable. Can be a Pandas or Polars Series.

        Returns:
            tuple: A tuple containing the transformed feature data (Xt) and the transformed target variable (yt).

        Raises:
            TypeError: If X or y are not Pandas or Polars DataFrames/Series when classic=True.

        Notes:
            - This method performs the following steps:
                1. Converts X and y to Polars DataFrames if they are Pandas DataFrames.
                2. Fits the feature selection pipeline and trains the model using `self.fit(X, y)`.
                3. Transforms the feature data using `self.transform(X)` to apply the feature selection.
                4. Transforms the target variable (y) using the `self.y_encoder` if the model type is classification.
                5. Fits the model to the transformed feature data and target variable: `self.model.fit(Xt[self.selected_features], yt)`.
                6. Sets `self.model_fitted_` to True to indicate that the model has been trained.
        """
        X = self._check_pandas(X)
        y = self._check_pandas(y)
        self.fit(X, y)
        Xt = self.transform(X)
        if self.model_type != 'regression':
            yt = self.y_encoder.transform(y)
        else:
            yt = y
        self.model.fit(Xt[self.selected_features], yt)
        self.model_fitted_ = True
        return Xt, yt

    def fit_predict(self, X, y):
        """
        Fits the Featurewiz_MRMR_Model to the input data and then makes predictions on the same data. This combines training and prediction for convenience.

        Args:
            X (pd.DataFrame or pl.DataFrame): The input feature data. Can be a Pandas or Polars DataFrame.
            y (pd.Series or pl.Series): The target variable. Can be a Pandas or Polars Series.

        Returns:
            np.ndarray: An array of predictions made by the trained model.

        Raises:
            ValueError: If the `model` argument was set to `None` during initialization. A model must be provided (either explicitly or by allowing the default model to be created) for predictions to be made.
            TypeError: If X or y are not Pandas or Polars DataFrames/Series when classic=True.

        Notes:
            - This method performs the following steps:
                1. Converts X and y to Polars DataFrames if they are Pandas DataFrames.
                2. Fits the feature selection pipeline and trains the model using `self.fit(X, y)`.
                3. Transforms the feature data using `self.transform(X)` to apply the feature selection.
                4. Transforms the target variable (y) using the `self.y_encoder` if the model type is classification.
                5. Fits the model to the transformed feature data and target variable.
                6. Makes predictions on the transformed feature data using the trained model: `self.model.predict(Xt[self.selected_features])`.
        """
        X = self._check_pandas(X)
        y = self._check_pandas(y)
        self.fit(X, y)
        if not self.model is None:
            Xt = self.transform(X)
            if self.model_type != 'regression':
                yt = self.y_encoder.transform(y)
            else:
                yt = y
            self.model.fit(Xt[self.selected_features], yt)
            self.model_fitted_ = True
            return self.model.predict(Xt[self.selected_features])
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
        if y is None:
            if self.model_fitted_:
                return self.model.predict(Xt[self.selected_features])
            else:
                print('Error: Model is not fitted yet. Please call fit_predict() first')
                return X
        else:
            if not self.model_fitted_:
                if self.model_type != 'regression':
                    yt = self.y_encoder.transform(y)
                else:
                    yt = y
                self.model.fit(Xt[self.selected_features], yt)
            return self.model.predict(Xt[self.selected_features])
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
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (array-like): The true (actual) values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The MAPE value, expressed as a percentage.

    Notes:
        - MAPE is a common metric for evaluating the accuracy of forecasting models.
        - The function handles potential division-by-zero errors by replacing zero values in `y_true` with 1, ensuring a stable calculation.
        - The formula used is: `mean(abs((y_true - y_pred) / max(1, abs(y_true)))) * 100`
    """    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100
################################################################################
def print_regression_metrics(y_true, y_preds, verbose=0):
    """
    Prints a comprehensive set of regression metrics to evaluate model performance.

    Args:
        y_true (array-like): The true (actual) values.
        y_preds (array-like): The predicted values.
        verbose (int, optional): Controls the level of output. If 1, prints detailed metrics. 
        If 0, prints a summary and generates a scatter plot. Defaults to 0.

    Returns:
        float: The Root Mean Squared Error (RMSE). Returns np.inf if an error occurs during calculation.

    Notes:
        - This function calculates and prints RMSE, Normalized RMSE, MAE, WAPE, Bias, MAPE, and R-Squared.
        - If `verbose` is set to 0, it generates a scatter plot of the true vs. predicted values using `plot_regression()`.
        - If there are zero values in `y_true`, it will print a warning that MAPE is not available and still calculates WAPE and Bias.
        - It handles potential exceptions during metric calculation and prints an error message if one occurs.
    """
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
                mape = 100*MAPE(y_true, y_preds)
                print('    MAPE = %0.0f%%' %(mape))
                if mape > 100:
                    print('\tHint: high MAPE: try np.log(y) instead of (y).')
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

