# polars_categorical_encoder_v2.py (Minor Clarity Tweaks)
import polars as pl
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
import copy
import pdb
from .polars_other_transformers import Polars_ColumnEncoder
#################################################################################################
class Polars_CategoricalEncoder(TransformerMixin): # Class name updated to V2
    """
    Encodes categorical features using Ordinal or Weight of Evidence (WoE) or Target Encoding, optimized for Polars DataFrames.
    Why our category encoder works better:
        - Polars-native Operations: Uses proper Series/DataFrame handling
        - Memory Efficiency: Maintains data in Polars space without converting to numpy
        - Batch Processing: Handles multiple columns while preserving individual encoders
        - Type Safety: Maintains consistent integer types across transformations

    Inputs:
    - encoding_type: can be "target", "woe" or "ordinal"
    - categorical_features = you can set it to "auto" or provide it explicit list of features you want handled
    - handle_unknown: must be either one of ['value', 'error']
    - unknown_value: must be None or float value.
    """
    def __init__(self, encoding_type='target', categorical_features='auto', handle_unknown='value', unknown_value=None,):
        self.encoding_type = encoding_type
        self.categorical_features = categorical_features
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categorical_feature_names_ = []

        if encoding_type not in ['woe', 'target', "ordinal"]:
            raise ValueError(f"Invalid encoding_type: '{encoding_type}'. Must be 'woe' or 'target' or 'ordinal'.")
        if handle_unknown not in ['value', 'error']:
            raise ValueError(f"Invalid handle_unknown: '{handle_unknown}'. Must be 'value' or 'error'.")
        if self.encoding_type == 'woe' and self.unknown_value is not None and not isinstance(self.unknown_value, float):
             raise ValueError(f"unknown_value for WoE encoding must be a float or None, got {type(self.unknown_value)}")


    def fit(self, X, y=None):
        """
        Fits the PolarsCategoricalFeatureEncoder to the data.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y (Polars Series or array-like): Target vector of shape (n_samples,) - Required for WoE and Target Encoding.

        Returns:
            self
        """
        
        if not isinstance(X, pl.DataFrame):
            raise ValueError("Input 'X' must be a Polars DataFrame.")

        if self.encoding_type == 'woe':
            check_classification_targets(y) # WOE is for classification

        if y is None:
            raise ValueError("Target 'y' must be provided for CategoricalFeatureEncoder.")
        if isinstance(y, pl.Series):
            y_pl = y
        else:
            y_pl = pl.Series(y)


        if self.categorical_features == 'auto':
            categorical_cols = [col for col in X.columns if X[col].dtype in [pl.Categorical, pl.Utf8]] # Detect String or Categorical columns
        else:
            categorical_cols = self.categorical_features
            for col in categorical_cols:
                if col not in X.columns:
                    raise ValueError(f"Your input categorical_features column '{col}' not found in your DataFrame.")


        self.categorical_feature_names_ = copy.deepcopy(categorical_cols)
        self.encoders_ = {} # Dictionary to store encoding mappings
        
        for feature in categorical_cols:
            if self.encoding_type == 'woe':
                if self.model_type != 'regresssion':
                    print('Weight of evidence encoding cannot be used in Regression. Please try using target encoding instead. Returning')
                    return self
                # Weight of Evidence Encoding (Polars Implementation)
                event_count = X.group_by(feature).agg(pl.count().alias('count'), pl.sum(pl.Series(y_pl) == 1).alias('event_count')) # Explicit pl.col()
                total_event = y_pl.sum()
                total_non_event = len(y_pl) - total_event

                woe_mapping = event_count.with_columns(
                    non_event_count = pl.col('count') - pl.col('event_count'),
                    event_rate = (pl.col('event_count') + 1e-9) / (total_event + 1e-7), # Add small epsilon to avoid div by zero and log(0)
                    non_event_rate = (pl.col('non_event_count') + 1e-9) / (total_non_event + 1e-7),
                    woe = (pl.col('event_rate') / pl.col('non_event_rate')).log()
                ).select([pl.col(feature), pl.col('woe')]).set_index(feature).to_dict()['woe'] # Explicit pl.col()

                self.encoders_[feature] = woe_mapping


            elif self.encoding_type == 'target':
                # Target Encoding - you need both X and y for target encoding and it can work for both model-types
                df = pl.concat([X, y.to_frame()], how='horizontal')
                dfx = df.group_by(feature).agg(pl.mean(y.name))  
                dfx = dfx.rename({y_pl.name:'target_mean'}) 
                target_mapping = dfx.to_pandas().set_index(feature).to_dict()['target_mean']
                #target_mapping = X.group_by(feature).agg(pl.mean(pl.Series(y_pl)).alias('target_mean')).set_index(feature).to_dict()['target_mean'] 
                self.encoders_[feature] = target_mapping


            elif self.encoding_type == 'ordinal':              
                # Create and fit individual encoder
                encoder = Polars_ColumnEncoder()
                encoder.fit(X.get_column(feature))
                
                # Store encoder with feature name as key
                self.encoders_[feature] = encoder


            else: # Should not happen due to init validation
                raise ValueError("Invalid encoding type (internal error).")


        self.fitted_ = True
        return self



    def transform(self, X, y=None):
        """
        Transforms the data by encoding categorical features using Polars operations.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)

        Returns:
            Polars DataFrame: Transformed DataFrame with encoded categorical features.
        """
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pl.DataFrame):
            raise ValueError("Input 'X' must be a Polars DataFrame for transform.")


        X_transformed = X.clone() # Create a copy to avoid modifying original DataFrame

        if self.encoding_type == 'ordinal':
            for feature, encoder in self.encoders_.items():
                # Get encoded values as polars Series
                encoded_series = pl.Series(
                    name=feature,
                    values=encoder.transform(X.get_column(feature)),
                    dtype=pl.Int32
                )
                
                # Replace existing column using lazy API
                X_transformed = X_transformed.with_columns(
                    encoded_series.alias(feature)
                )
        else:
            for feature in self.categorical_feature_names_:
                if feature in self.encoders_:
                    encoding_map = self.encoders_[feature]
                    if self.handle_unknown == 'value':
                        unknown_val = self.unknown_value if self.unknown_value is not None else np.nan # Default unknown value to NaN if None provided
                        X_transformed = X_transformed.with_columns(pl.col(feature).replace(encoding_map, default=unknown_val).alias(feature))

                    elif self.handle_unknown == 'error':
                        if any(cat not in encoding_map for cat in X_transformed[feature].unique()): # Check for unknown categories
                            unknown_categories = [cat for cat in X_transformed[feature].unique() if cat not in encoding_map]
                            raise ValueError(f"Unknown categories '{unknown_categories}' encountered in feature '{feature}' during transform.")
                        X_transformed = X_transformed.with_columns(pl.col(feature).replace(encoding_map).alias(feature))
                else:
                    # Should ideally not reach here if fit and transform are used correctly, but for robustness:
                    if self.handle_unknown == 'value':
                        X_transformed = X_transformed.with_columns(pl.lit(self.unknown_value).alias(feature)) # Fill with unknown value
                    elif self.handle_unknown == 'error':
                        raise ValueError(f"Feature '{feature}' was specified as categorical but not seen during fit.")


        if y is None:
            return X_transformed
        else:
            return X_transformed, y # Return as numpy array if requested


    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        For PolarsCategoricalFeatureEncoder, output feature names are the same as input categorical feature names.
        """
        check_is_fitted(self, 'fitted_')
        return self.categorical_feature_names_ # Encoded features retain original names in this implementation





