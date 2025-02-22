import numpy as np
np.random.seed(42)
import random
random.seed(42)
import polars as pl
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from xgboost import XGBClassifier, XGBRegressor
from typing import List, Dict
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import copy
import pdb
from collections import defaultdict
###########################################################################################
class Polars_MissingTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for filling null and NaN values in Polars DataFrames.
    Handles "float" columns only. Ignores integers and categoricals which are handled by another transformer.
    
    Parameters:
    strategy (str): One of ["zeros", "mean", "median"], default="zeros"
    
    Usage:
    transformer = Polars_MissingFiller(strategy="mean")
    df_filled = transformer.fit_transform(df)
    """
    def __init__(self, strategy="zeros"):
        assert strategy in ["zeros", "mean", "median"], \
            "Invalid strategy. Choose from: zeros, mean, median"
        self.strategy = strategy
        self.fill_values_ = None
        self.float_columns_ = []

    def fit(self, X: pl.DataFrame, y=None):
        # Identify float columns
        self.float_columns_ = [
            col for col in X.columns 
            if X[col].dtype in (pl.Float32, pl.Float64)
        ]
        
        self.fill_values_ = {}
        if self.strategy in ("mean", "median"):
            
            for col in self.float_columns_:
                # Handle both NaN and null values in calculations
                clean_series = X[col].fill_nan(None)
                
                if self.strategy == "mean":
                    value = clean_series.mean()
                else:
                    value = clean_series.median()
                
                # Fallback to 0 if all values are NaN/null
                self.fill_values_[col] = value if value is not None else 0.0
        else:
            for col in self.float_columns_:
                self.fill_values_[col] = 0.0

        self.fitted_ = True
        return self

    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        check_is_fitted(self, 'fitted_')
        X_transformed = X.clone()
        
        for col in self.float_columns_:
            if col not in X.columns:
                continue  # Skip columns removed since fitting
                
            fill_value = self.fill_values_.get(col, 0.0)

            # Fill both NaN and null values
            X_transformed = X_transformed.with_columns(
                pl.col(col)
                .fill_nan(fill_value)
                .fill_null(fill_value)
            )
            
        return X_transformed

    def fit_transform(self, X, y=None):
        try:
            check_is_fitted(self, 'fitted_')
            return self.transform(X, y) 
        except:
            return self.fit(X, y).transform(X, y)

    def get_feature_names_out(self, input_features=None):
        return self.float_columns_
#####################################################################################################
class Polars_ColumnEncoder(TransformerMixin):
    """
    Polars-based Ordinal Encoder that works on only one column at time
    - you have to call it repeatedly if you want to convert multiple columns
    - the good news is that it handles nulls and unseen values in each column using Polars
    Features:
    - Automatically handles null/nan values as distinct category in a column
    - Assigns new codes to unseen values during transform in a column
    - Compatible with scikit-learn pipelines as long as you use it per column with X as a Series

    Input:
    X - must be a Polars Series 
    y is returned as is with  no changes
    """
    
    def __init__(self):
        self.transformer = dict()
        self.inverse_transformer = dict()
        self.max_val = 0

    def _handle_input(self, testx):
        """Convert various input types to polars Series"""
        if isinstance(testx, tuple):
            _, testx = testx  # Discard y if present
            
        if isinstance(testx, pl.DataFrame):
            if testx.shape[1] == 1:
                return testx.to_series()
            return testx
        elif isinstance(testx, (pd.Series, np.ndarray)):
            return pl.Series(testx)
        elif isinstance(testx, pl.Series):
            return testx
        else:
            try:
                return pl.Series(testx)
            except Exception as e:
                raise ValueError(f"Unsupported input type: {type(testx)}") from e

    def fit(self, X, y=None):
        X = self._handle_input(X)
        
        if isinstance(X, pl.DataFrame):
            if X.shape[1] != 1:
                return self
            X = X.to_series()

        # Handle empty case
        if X.is_empty():
            return self

        unique_values = X.unique().to_list()
        codes = list(range(len(unique_values)))
        
        self.transformer = {val: code for val, code in zip(unique_values, codes)}
        self.inverse_transformer = {code: val for val, code in zip(unique_values, codes)}
        self.max_val = max(codes, default=0)
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        X = self._handle_input(X)
        
        if isinstance(X, pl.DataFrame):
            if X.shape[1] != 1:
                return X.to_numpy()
            X = X.to_series()

        # Handle empty case
        if X.is_empty():
            return np.array([])

        # Identify new values
        new_values = X.unique().to_list()
        missing = [val for val in new_values if val not in self.transformer]

        # Update encoding dictionaries
        for val in missing:
            self.max_val += 1
            self.transformer[val] = self.max_val
            self.inverse_transformer[self.max_val] = val

        # Perform the encoding
        encoded = X.replace(self.transformer).cast(pl.Int32)
        
        if y is None:
            return encoded
        else:
            return encoded, y

    def fit_transform(self, X, y=None):
        try:
            check_is_fitted(self, 'fitted_')
        except:
            return self.fit(X, y).transform(X, y)
        return self.transform(X, y) 

    def inverse_transform(self, X, y=None):
        if not isinstance(X, (pl.Series, np.ndarray, list)):
            raise ValueError("Input must be array-like")
            
        X = pl.Series(X)
        decoded = X.replace(self.inverse_transformer)
        return decoded.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{input_features[0]}_encoded"]
###############################################################################################
# This is needed to make this a regular transformer ###
class YTransformer(TransformerMixin):
    """
    Polars-based "y" transformer that works on only one column in the "y" target variable
    - you have to call it repeatedly if you want to convert multiple columns in "y"
    - the good news is that it handles nulls and unseen values in "y" using Polars
    Features:
    - Automatically handles null/nan values as distinct category in a column
    - Assigns new codes to unseen values during transform in a column
    - Compatible with scikit-learn pipelines as long as you use it per column with X as a Series

    Input:
    X - is returned as is with  no changes
    y - must be a Polars Series  - otherwise it will return it as is
    """
    
    def __init__(self):
        self.transformer = dict()
        self.inverse_transformer = dict()
        self.max_val = 0

    def _handle_input(self, testx):
        """Convert various input types to polars Series"""
        if isinstance(testx, tuple):
            _, testx = testx  # Discard y if present
            
        if isinstance(testx, pl.DataFrame):
            if testx.shape[1] == 1:
                return testx.to_series()
            return testx
        elif isinstance(testx, (pd.Series, np.ndarray)):
            return pl.Series(testx)
        elif isinstance(testx, pl.Series):
            return testx
        else:
            try:
                return pl.Series(testx)
            except Exception as e:
                raise ValueError(f"Unsupported input type: {type(testx)}") from e

    def fit(self, X, y=None):
        if y is None:
            return self

        y = self._handle_input(y)
        
        if isinstance(y, pl.DataFrame):
            if y.shape[1] != 1:
                return self
            y = y.to_series()

        # Handle empty case
        if y.is_empty():
            return self

        unique_values = y.unique().to_list()
        codes = list(range(len(unique_values)))
        
        self.transformer = {val: code for val, code in zip(unique_values, codes)}
        self.inverse_transformer = {code: val for val, code in zip(unique_values, codes)}
        self.max_val = max(codes, default=0)
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        if y is None:
            return X

        y = self._handle_input(y)
        
        if isinstance(y, pl.DataFrame):
            if y.shape[1] != 1:
                return y.to_numpy()
            y = y.to_series()

        # Handle empty case
        if y.is_empty():
            return np.array([])

        # Identify new values
        new_values = y.unique().to_list()
        missing = [val for val in new_values if val not in self.transformer]

        # Update encoding dictionaries
        for val in missing:
            self.max_val += 1
            self.transformer[val] = self.max_val
            self.inverse_transformer[self.max_val] = val

        # Perform the encoding
        encoded = y.replace(self.transformer).cast(pl.Int32)

        if y is None:
            return X
        else:
            return X, encoded

    def fit_transform(self, X, y=None):
        try:
            check_is_fitted(self, 'fitted_')
            return self.transform(X, y) 
        except:
            ### if it is not fitted, you must fit and then call it ##
            return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        if y is None:
            return X

        if not isinstance(y, (pl.Series, np.ndarray, list)):
            raise ValueError("y Input must be Series or array-like")
            
        y = pl.Series(y)
        decoded = y.replace(self.inverse_transformer)
        return X, decoded.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return [f"{input_features[0]}_encoded"]

##############################################################################