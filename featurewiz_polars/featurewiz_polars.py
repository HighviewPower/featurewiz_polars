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
# Needed imports
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # Or LightGBM, XGBoost, etc.
# Import the Polars classes now
from .polars_categorical_encoder import Polars_CategoricalEncoder # Now using V2 of Encoder
from .polars_datetime_transformer import Polars_DateTimeTransformer # Import new date-time transformer
from .polars_other_transformers import YTransformer, Polars_MissingTransformer, Polars_ColumnEncoder
from .polars_sulov_mrmr import Sulov_MRMR
import time
import copy
import pdb
#############################################################################
class Featurewiz_MRMR(BaseEstimator, TransformerMixin): # Class name 
    def __init__(self, 
            model_type='classification', encoding_type='target', 
            imputation_strategy='mean', corr_threshold = 0.7,
            classic = False, estimator=None,
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

            estimator (estimator object, optional): If None, it implements the original XGBoost approach with recursive feature elimination.
            Your options are: randomforest or catboost or lightgbm models for feature selection. Defaults to None.

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
        self.estimator = estimator
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
                    ('featurewiz', Sulov_MRMR(corr_threshold=self.corr_threshold, estimator=self.estimator,
                    model_type=self.model_type, classic=self.classic, verbose=self.verbose)),
                ])

        feature_selection = Pipeline([
                ('PreProcessing_pipeline', preprocessing_pipeline),
                ('Featurewiz_pipeline', featurewiz_pipeline)
            ])

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder
        self.estimator_name = self._get_model_name(estimator)

    def _get_model_name(self, mod):
        if 'catboost' in str(mod).lower():
            return 'CatBoost'
        elif 'lgbm' in str(mod).lower():
            return 'LightGBM'
        elif 'randomforest' in str(mod).lower():
            return "RandomForest"
        else:
            return 'XGBoost'

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
        if self.model_type != 'regression':
            self.y_encoder.fit(y)

        ### If successful save all the pipelines in following variables to use later in transform and predict
        self.preprocessing_pipeline = self.feature_selection[-2]
        self.featurewiz_pipeline = self.feature_selection[-1]
        ### since this is a pipeline within a pipeline, you have to use nested lists to get the features!
        self.selected_features = self.feature_selection[-1][-1].get_feature_names_out()
        print(f'\nFeaturewiz-Polars feature selection with {self.estimator_name} estimator completed.')
        print('Time taken  = %0.1f seconds' %(time.time()-start_time))
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
            return self.feature_selection.transform(X)[self.selected_features]
        else:
            Xt = self.feature_selection.transform(X)[self.selected_features]
            if self.model_type != 'regression':
                yt = self.y_encoder.transform(y)
            else:
                yt = copy.deepcopy(y)
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
        Xt = self.transform(X)[self.selected_features]
        if self.model_type != 'regression':
            yt = self.y_encoder.transform(y)
        else:
            yt = copy.deepcopy(y)
        return Xt, yt

##############################################################################
class Featurewiz_MRMR_Model(BaseEstimator, TransformerMixin): # Class name 
    """
    Initializes the Featurewiz_MRMR_Model class for feature engineering, selection, and model training.

    Args:
    *   **`estimator`**  (estimator object, *optional*): This argument is used to by featurewiz to do the feature selection. 
        Only the following model estimators are supported: XGBoost, CatBoost, RandomForest and LightGBM 

    *   **`model`** (estimator object, *optional*): This estimator is used in the pipeline to train a new model `after feature selection`.
        If `None`, a default estimator (Random Forest) will be trained after selection. Defaults to `None`. 
        This `model` argument can be different from the `estimator` argument above.
        Only the following model estimators are supported: XGBoost, CatBoost, RandomForest and LightGBM 

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
        classic=False, estimator=None,
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
        self.estimator = estimator
        # MRMR is same for regression and classification
        feature_selection = Featurewiz_MRMR(model_type=self.model_type, 
            encoding_type=self.encoding_type, 
            imputation_strategy=self.imputation_strategy, 
            corr_threshold =self.corr_threshold,
            classic=self.classic, estimator=self.estimator,
            verbose=self.verbose)

        ### You need to separately create a column encoder because you will need this for transforming y_test later!
        y_encoder = Polars_ColumnEncoder()
        self.feature_selection = feature_selection
        self.y_encoder = y_encoder
        self.model_name = self._get_model_name(model)

    def _get_model_name(self, mod):
        if 'catboost' in str(mod).lower():
            return 'CatBoost'
        elif 'lgbm' in str(mod).lower():
            return 'LightGBM'
        elif 'randomforest' in str(mod).lower():
            return "RandomForest"
        else:
            return 'XGBoost'

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
        print(f'\nFeaturewiz-Polars MRMR completed training with {self.model_name} Model.')
        print('Total time taken  = %0.1f seconds' %(time.time()-start_time))
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
            self.model = self._fit_different_models(Xt[self.selected_features], yt)
            ### The model is fitted now so self.model_fitted_ is set to True
            self.model_fitted_ = True
            return Xt, yt

    def _fit_different_models(self, X, y):
        if 'catboost' in str(self.model).lower():
            self.model.fit(X.to_pandas(), y.to_pandas())
        elif 'lgbm' in str(self.model).lower():
            self.model.fit(X.to_pandas(), y.to_pandas(), categorical_feature='auto', 
                    feature_name='auto')
        else:
            self.model.fit(X, y)
        return self.model

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
        self.model = self._fit_different_models(Xt, yt)
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
            self.model = self._fit_different_models(Xt[self.selected_features], yt)
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
                if 'catboost' in str(self.model).lower():
                    return self.model.predict(Xt.to_pandas())
                elif 'lgbm' in str(self.model).lower():
                    return self.model.predict(Xt.to_pandas())
                else:
                    return self.model.predict(Xt)
            else:
                print('Error: Model is not fitted yet. Please call fit_predict() first')
                return X
        else:
            if not self.model_fitted_:
                if self.model_type != 'regression':
                    yt = self.y_encoder.transform(y)
                else:
                    yt = y
                ### Now fit the model and predict since it is not fitted yet ###
                self.model = self._fit_different_models(Xt, yt)
                if 'catboost' in str(self.model).lower():
                    return self.model.predict(Xt.to_pandas())
                elif 'lgbm' in str(self.model).lower():
                    return self.model.predict(Xt.to_pandas())
                else:
                    return self.model.predict(Xt)

    def predict_proba(self, X, y=None) :
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
                if self.model_type != 'regression':
                    if 'catboost' in str(self.model).lower():
                        return self.model.predict_proba(Xt[self.selected_features].to_pandas())
                    else:
                        return self.model.predict_proba(Xt[self.selected_features])
            else:
                print('Error: Model is not fitted yet. Please call fit_predict() first')
                return Xt
        else:
            if not self.model_fitted_:
                if self.model_type != 'regression':
                    yt = self.y_encoder.transform(y)
                else:
                    yt = y
                self.model = self._fit_different_models(Xt[self.selected_features], yt)
            if 'catboost' in str(self.model).lower():
                return self.model.predict_proba(Xt[self.selected_features].to_pandas())
            elif 'lgbm' in str(self.model).lower():
                return self.model.predict_proba(Xt[self.selected_features].to_pandas())
            else:
                return self.model.predict_proba(Xt[self.selected_features])
##############################################################################
