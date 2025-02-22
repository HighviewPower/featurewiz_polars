# polars_datetime_transformer.py
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import polars.selectors as cs
#############################################################################
class Polars_DateTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts date and time features from Polars DataFrame columns.

    Parameters:
        datetime_features (list or 'auto', default='auto'):
            List of column names to treat as datetime features.
            If 'auto', infers datetime features based on Polars dtype `Datetime`.
        feature_prefix (str, default='dt_'):
            Prefix to add to generated datetime feature names.

    Attributes:
        datetime_feature_names_ (list): List of names of datetime features after fitting.
    """
    def __init__(self, datetime_features='auto', feature_prefix='dt_'):
        self.datetime_features = datetime_features
        self.datetime_feature_names_ = []
        self.feature_prefix = feature_prefix

    def fit(self, X, y=None):
        """
        Fits the PolarsDateTimeFeatureTransformer (identifies datetime columns).

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)

        Returns:
            self
        """
        if not isinstance(X, pl.DataFrame):
            raise ValueError("Input 'X' must be a Polars DataFrame.")

        if self.datetime_features == 'auto':
            datetime_cols = [col for col in X.columns if X[col].dtype == pl.Datetime] # Detect Datetime columns
            if len(datetime_cols) == 0:
                datetime_cols = X.select(cs.temporal()).columns
        else:
            datetime_cols = self.datetime_features
            for col in datetime_cols:
                if col not in X.columns:
                    raise ValueError(f"Specified datetime_features column '{col}' not found in input DataFrame columns.")

        self.datetime_feature_names_ = datetime_cols
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Transforms the data by extracting date and time features using Polars operations.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)

        Returns:
            Polars DataFrame: Transformed DataFrame with extracted datetime features appended.
        """
        check_is_fitted(self, 'fitted_')
        if not isinstance(X, pl.DataFrame):
            raise ValueError("Input 'X' must be a Polars DataFrame for transform.")

        X_transformed = X.clone() # Create a copy

        for feature in self.datetime_feature_names_:
            prefix = self.feature_prefix + feature + "_"
            X_transformed = X_transformed.with_columns([
                pl.col(feature).dt.year().alias(prefix + 'year'),
                pl.col(feature).dt.month().alias(prefix + 'month'),
                pl.col(feature).dt.day().alias(prefix + 'day'),
                #pl.col(feature).dt.hour().alias(prefix + 'hour'),
                pl.col(feature).dt.weekday().alias(prefix + 'weekday'), # Monday=0, Sunday=6
                pl.col(feature).dt.ordinal_day().alias(prefix + 'ordinal_day'), # Day of year (1-366)
            ])
        if y is None:
            return X_transformed
        else:
            return X_transformed, y

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        """
        check_is_fitted(self, 'fitted_')
        output_feature_names = []
        for feature in self.datetime_feature_names_:
            prefix = self.feature_prefix + feature + "_"
            output_feature_names.extend([
                prefix + 'year',
                prefix + 'month',
                prefix + 'day',
                #prefix + 'hour',
                prefix + 'weekday',
                prefix + 'ordinal_day'
            ])
        return output_feature_names

