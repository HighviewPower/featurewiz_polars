# ./tests/test_featurewiz.py (Improved Version)

import pytest
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
import numpy as np
from sklearn import datasets

# Import components from your library
from featurewiz_polars import FeatureWiz, polars_train_test_split

# --- Fixtures for Scikit-learn Datasets ---

@pytest.fixture(scope="module") # Use module scope for efficiency - load data once per module
def breast_cancer_data():
    """Loads breast cancer dataset and returns as a Polars DataFrame."""
    cancer = datasets.load_breast_cancer(as_frame=True)
    df_pandas = cancer.frame
    # Combine features and target into one Polars DataFrame
    df_polars = pl.from_pandas(df_pandas)
    # Ensure target is integer for classification
    df_polars = df_polars.with_columns(pl.col("target").cast(pl.Int64))
    return df_polars, "target"

@pytest.fixture(scope="module")
def diabetes_data():
    """Loads diabetes dataset and returns as a Polars DataFrame."""
    diabetes = datasets.load_diabetes(as_frame=True)
    df_pandas = diabetes.frame
    # Combine features and target into one Polars DataFrame
    df_polars = pl.from_pandas(df_pandas)
    # Target is already float
    return df_polars, "target"

# --- Test Cases ---

def test_initialization_defaults():
    """Test FeatureWiz initialization with default parameters (implicitly)."""
    # Note: README example initializes with explicit args,
    # testing defaults requires knowing them or initializing without args if possible.
    # Assuming target_variable is mandatory based on previous examples.
    # If FeatureWiz has defaults allowing initialization without target, test that.
    # Based on README, let's assume target is needed, maybe model_type etc. have defaults.
    try:
        # Initialize based on README's example structure:
        fwiz = FeatureWiz(model_type="Classification",  verbose=0)
        assert fwiz.model_type.lower() == "classification"
        assert fwiz.corr_limit == 0.7 # Assuming default from README example
        assert fwiz.category_encoders == 'target' # Assuming default is target encoding
        # Add assertions for other important default attributes
    except TypeError as e:
        # This handles cases where more arguments are required than assumed
        pytest.fail(f"FeatureWiz initialization failed with defaults: {e}")


def test_classification(breast_cancer_data):
    """Tests the full Classification workflow using breast cancer data."""
    df, target_name = breast_cancer_data
    predictors = [col for col in df.columns if col != target_name]

    X = df.select(predictors)
    y = df.select(target_name) # Keep as DataFrame column for split function

    # 1. Split data using the library's function
    X_train, X_test, y_train, y_test = polars_train_test_split(
        X, y, test_size=0.2, random_state=42, 
    )
    # Extract Series after split for FeatureWiz API
    y_train_series = y_train[target_name]
    y_test_series = y_test[target_name]


    # 2. Initialize FeatureWiz (matching README example)
    fwiz = FeatureWiz(
        model_type="Classification",
        estimator=None,
        corr_limit=0.7,
        category_encoders='onehot',
        classic=True, # Keep or remove based on its actual meaning/necessity
        verbose=0
    )

    # 3. Fit and Transform Training Data
    X_train_transformed, y_train_transformed = fwiz.fit_transform(X_train, y_train_series)

    # Assertions for fit_transform
    assert isinstance(X_train_transformed, pl.DataFrame)
    assert isinstance(y_train_transformed, pl.Series) # Check if target is transformed/returned correctly
    assert target_name not in X_train_transformed.columns # Target shouldn't be in features
    assert len(fwiz.selected_features) > 0 # Should select at least some features
    assert len(fwiz.selected_features) <= len(predictors) # Cannot select more than original
    assert set(X_train_transformed.columns) == set(fwiz.selected_features)
    assert X_train_transformed.shape[0] == X_train.shape[0] # Number of rows should match
    # Check if y_train was encoded (depends on FeatureWiz logic)
    if hasattr(fwiz, 'y_encoder') and fwiz.y_encoder:
        assert y_train_transformed.dtype != y_train_series.dtype # Example check if encoding happened
        assert len(np.unique(y_train_transformed)) == len(np.unique(y_train_series))
    else:
         assert_series_equal(y_train_transformed, y_train_series) # Should be unchanged if no encoding


    # 4. Transform Test Data
    X_test_transformed = fwiz.transform(X_test)

    # Assertions for transform
    assert isinstance(X_test_transformed, pl.DataFrame)
    assert X_test_transformed.shape[0] == X_test.shape[0] # Number of rows should match
    assert list(X_test_transformed.columns) == list(X_train_transformed.columns) # Columns must match train transform
    assert len(X_test_transformed.columns) == len(fwiz.selected_features)

    # 5. Transform Test Target (if applicable)
    if hasattr(fwiz, 'y_encoder') and fwiz.y_encoder:
        y_test_transformed = fwiz.y_encoder.transform(y_test_series)
        assert isinstance(y_test_transformed, pl.Series) # Or numpy array depending on encoder
        assert y_test_transformed.shape[0] == y_test_series.shape[0]
    else:
        # If no encoding, check consistency or maybe this step isn't needed
        pass # Or assert y_test is unchanged by featurewiz itself


def test_regression(diabetes_data):
    """Tests the full Regression workflow using diabetes data."""
    df, target_name = diabetes_data
    predictors = [col for col in df.columns if col != target_name]

    X = df.select(predictors)
    y = df.select(target_name) # Keep as DataFrame for split

    # 1. Split data
    X_train, X_test, y_train, y_test = polars_train_test_split(
        X, y, test_size=0.25, random_state=123, 
    )
    y_train_series = y_train[target_name]
    y_test_series = y_test[target_name]

    # 2. Initialize FeatureWiz for Regression
    fwiz = FeatureWiz(
        model_type="Regression", # Specify Regression
        estimator=None,
        corr_limit=0.8, # Use a different limit for variety
        category_encoders='onehot', # Assuming onehot is the default
        classic=True, # Keep or remove based on its actual meaning/necessity
        verbose=0
    )

    # 3. Fit and Transform Training Data
    X_train_transformed, y_train_transformed = fwiz.fit_transform(X_train, y_train_series)

    # Assertions for fit_transform
    assert isinstance(X_train_transformed, pl.DataFrame)
    assert isinstance(y_train_transformed, pl.Series)
    assert target_name not in X_train_transformed.columns
    assert len(fwiz.selected_features) > 0
    assert len(fwiz.selected_features) <= len(predictors)
    assert set(X_train_transformed.columns) == set(fwiz.selected_features)
    assert X_train_transformed.shape[0] == X_train.shape[0]
    # Regression targets are typically not encoded, assert they are equal
    assert_series_equal(y_train_transformed, y_train_series)

    # 4. Transform Test Data
    X_test_transformed = fwiz.transform(X_test)

    # Assertions for transform
    assert isinstance(X_test_transformed, pl.DataFrame)
    assert X_test_transformed.shape[0] == X_test.shape[0]
    assert list(X_test_transformed.columns) == list(X_train_transformed.columns)

    # 5. Transform Test Target (Usually not needed for Regression targets)
    # No y_encoder expected for regression targets in typical scenarios.

# Add more tests:
# - Test different `category_encoders` (if you add data with actual categorical features).
# - Test specific estimators if the `estimator` parameter is used for selection.
# - Test edge cases: DataFrames with NaNs, zero variance columns, etc.
# - Test error handling: Pass invalid `model_type`, non-existent target, etc. (already have some)
# - Test `polars_train_test_split` itself more directly if needed.