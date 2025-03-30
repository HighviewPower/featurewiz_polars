![featurewiz_polars_logo](./images/featurewiz_polars_taking_off.jpg)
# Featurewiz-Polars 🚀

[![PyPI version](https://img.shields.io/pypi/v/featurewiz_polars.svg)](https://pypi.org/project/featurewiz_polars/)
[![License: Apache2.0](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://opensource.org/licenses/Apache2.0)

**Fast, Automated Feature Engineering and Selection using Polars!**

`featurewiz_polars` is a high-performance Python library designed to accelerate your machine learning workflows by automatically creating and selecting the best features from your dataset. It leverages the speed and memory efficiency of the [Polars](https://www.pola.rs/) DataFrame library.

## ✨ Quick Start

Get started in minutes! Here's a minimal example to create some mock data:

```python
import polars as pl

# Create a sample Polars DataFrame
data = {
    'col1': [1, 2, 1, 3, 4, 5, 1, 6],
    'col2': [10.0, 11.5, 10.0, 12.5, 13.0, 14.5, 10.0, 15.0],
    'category': ['A', 'B', 'A', 'B', 'C', 'A', 'A', 'C'],
    'target': [0, 1, 0, 1, 1, 0, 0, 1]
}
df = pl.DataFrame(data)
```

Or you can load a CSV file into `polars` library's dataframes for use with `featurewiz-polars`. Use this code snippet exclusively for `featurewiz-polars` pipelines.

```python
# Load a CSV file into Polars DataFrames using:

import polars as pl
df = pl.read_csv(datapath+filename, null_values=['NULL','NA'], try_parse_dates=True,
    infer_schema_length=10000, ignore_errors=True)

# Before we do feature selection we always need to make sure we split the data #######
target = 'target'
predictors = [x for x in df.columns if x not in [target]]

X = df[predictors]
y = df[target] 

# BEWARE WHEN USING SCIKIT-LEARN `train_test_split` WITH POLARS DATA FRAMES!
# If you perform train-test split using sklearn it will give different train test rows each time
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instead you must split using polars_train_test_split with seed parameter 
# This will ensure your train-test splits are same each time to get same rows of data

from featurewiz_polars import polars_train_test_split
X_train, X_test, y_train, y_test = polars_train_test_split(X, y, test_size=0.2, random_state=42)
```

Once you have performed train_test_split on your Polars dataframe, you can initialize featurewiz_polars and perform feature engineering and selection:

```python
from featurewiz_polars import FeatureWiz

# Initialize FeatureWiz for classification
wiz = FeatureWiz(model_type="Classification", estimator=None,
        corr_limit=0.7, category_encoders='onehot', classic=True, verbose=0)

# Fit and transform the training data
X_transformed, y_transformed = wiz.fit_transform(X_train, y_train)

# Transform the test data
X_test_transformed = wiz.transform(X_test)

# Transform the test target variable
y_test_transformed = wiz.y_encoder.transform(y_test)
```

Now you can display the selected features and use them further in your model training pipelines:
```python
# View results
print("Selected Features:")
print(wiz.selected_features)
# Example Output: ['col1', 'col2', 'category_A', 'category_B']

print("\nTransformed DataFrame head:")
print(X_transformed.head())
# Example Output: Polars DataFrame with only the selected features
```

## 🤔 Why Use featurewiz_polars?
While there are many tools for feature manipulation, featurewiz_polars offers a unique combination of speed, automation, and specific algorithms:

**Vs. Original featurewiz (Pandas):**

**Speed & Memory:** Built entirely on Polars, `featurewiz_polars` offers exceptional speed and memory efficiency. It is particularly well-suited for handling datasets that exceed the limits of Pandas, leveraging Polars' multi-threaded processing and highly optimized Rust-based backend for superior performance.

### 🚀 Why Choose featurewiz_polars?

**Modern Backend:** Harnesses the power of the cutting-edge Polars ecosystem for unparalleled speed and efficiency.

**Vs. scikit-learn Preprocessing/Selection:**

- **Seamless Automation:** Combines feature engineering (e.g., interactions, group-by features) and feature selection into a single, streamlined pipeline. Unlike scikit-learn, which often requires manual configuration of multiple transformers, `featurewiz_polars` simplifies the process.
- **SULOV Algorithm:** Features the "Searching for Uncorrelated List of Variables" (SULOV) method—a fast, effective approach to identifying a diverse set of predictive features. This often results in simpler, more robust models. While scikit-learn offers methods like RFE, SelectKBest, and L1-based selection, SULOV is a unique advantage.
- **Integrated Workflow:** Transforms raw data into a model-ready feature set with minimal effort, making it ideal for end-to-end machine learning pipelines.

**When to Use featurewiz_polars:**

- **Handle Large Datasets:** Designed for maximum performance on datasets that push the limits of traditional tools.
- **Automate Feature Engineering:** Save time with automated creation and selection of impactful features.
- **Leverage Advanced Techniques:** Unlock the power of SULOV and other specialized algorithms for superior feature selection.

With `featurewiz_polars`, you get speed, simplicity, and cutting-edge techniques—all in one package.

## 💾 Installation: How to Install featurewiz_polars?
Install featurewiz_polars directly from PyPI:

```
pip install featurewiz_polars
```

Or, install the latest development version directly from GitHub:

```
pip install git+https://github.com/AutoViML/featurewiz_polars.git
```

<h3>Tests</h3>

Unlike a new library, the `featurewiz-polars` library is well-tested. To help you get started, I've provided example scripts like `fs_test.py`. This script demonstrates how to unit-test the library using two existing datasets in ./data sub folder in a concise manner. 

To run the unit tests, simply run the following command in your terminal:
```
cd tests
python fs_test.py
```

Additionally, I have provided two additional scripts for benchmarking. The `fs_lazytransform_test.py` script allows you to compare the performance of `featurewiz-polars` after feature engineering with the `lazytransform` library. This is a great boon for users who want to effortlessly create hundreds of features using the lazytransform library and then use featurew-polars for feature selection. 
```
cd tests
python fs_lazytransform_test.py
```

For a more in-depth benchmarking comparison between this library and another popular MRMR library, use `fs_mr_comparison_test.py` script. This script compares the performance of `featurewiz-polars` with another MRMR library.
```
cd tests
python fs_lazytransform_test.py
```

If you prefer working in a Jupyter Notebook or Colab, here are direct links to work in Colab with featurewiz-polars:

<h3>Examples</h3>

I have provided additional examples in ./examples sub folder. Anybody can open a copy of my Github-hosted notebooks within Colab. To make it easier I have created `Open-in-Colab` links to those GitHub-hosted notebooks below:

<h4>Featurewiz-Polars Test Notebook</h4>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AutoViML/featurewiz_polars/blob/main/examples/featurewiz-polars-test.ipynb)

<h4>Featurewiz-Polars vs classic featurewiz comparison test</h4>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AutoViML/featurewiz_polars/blob/main/examples/fw_polars_vs_featurewiz_test.ipynb)


## Feature Selection with `featurewiz-polars`

### Two Approaches

### 1. Feature Selection Only with `FeatureWiz` transformer

You have already seen this in the Quick Start section.

### 2. Feature Selection and Model Training with `FeatureWiz_Model` pipeline

This approach combines both feature selection and model training into a single scikit-learn pipeline.

```python
from featurewiz_polars import FeatureWiz_Model
from xgboost import XGBRegressor

# Initialize FeatureWiz_Model for regression with an XGBoost Regressor
wiz_model = FeatureWiz_Model(model_type="Regression", model=XGBRegressor(),
            corr_limit=0.7, category_encoders='onehot', classic=True, verbose=0)

# Fit and transform the training data
X_transformed, y_transformed = wiz_model.fit_transform(X_train, y_train)

# Make predictions on test data
y_pred = wiz_model.predict(X_test)

# View results
print(wiz_model.selected_features)
```

**Key Points:**

*   We use the `FeatureWiz_Model` class to combine feature selection and model training.
*   The `fit_transform` method is used to fit the feature selection process *and* train the specified model on the training data.
*   The `predict` method handles both transforming the test data using the learned feature selection and making predictions with the trained model, streamlining the entire process.

<h3>Arguments for featurewiz_polars Pipeline</h3>

The `FeatureWiz` class and `FeatureWiz_Model` class are designed for building data pipelines that use the feature engineering, selection, and model training capabilities of Polars. All you need to do is to upload your data into Polars DataFrames and then start calling these pipelines.

#### Arguments for pipelines

*   **`estimator`**  (estimator object, *optional*): This argument is used to by featurewiz to do the feature selection. 
        Only the following model estimators are supported: XGBoost, CatBoost, RandomForest and LightGBM 

*   **`model`** (estimator object, *optional*): This estimator is used in the pipeline to train a new model `after feature selection`.
        If `None`, a default estimator (Random Forest) will be trained after selection. Defaults to `None`. 
        This `model` argument can be different from the `estimator` argument above.
        Only the following model estimators are supported: XGBoost, CatBoost, RandomForest and LightGBM 

*   **`model_type`** (str, *optional*): The type of model to be built (`'classification'` or `'regression'`). Determines the appropriate preprocessing and feature selection strategies. Defaults to `'classification'`.

*   **`category_encoders`** (str, *optional*): The type of encoding to apply to categorical features (`'target'`, `'onehot'`, etc.).  `'woe'` encoding is only available for classification model types. Defaults to `'target'`.

*   **`imputation_strategy`** (str, *optional*): The strategy for handling missing values (`'mean'`, `'median'`, `'zeros'`). Determines how missing data will be filled in before feature selection. Defaults to `'mean'`.

*   **`corr_limit`** (float, *optional*): The correlation threshold for removing highly correlated features. Features with a correlation above this threshold will be targeted for removal. Defaults to `0.7`.

*   **`classic`** (bool, *optional*): If `True`, it implements the original classic `featurewiz` library using Polars. If `False`, implements the train-validation-split-recursive-xgboost version, which is faster and uses train/validation splits to stabilize features. Defaults to `False`.

*   **`verbose`** (int, *optional*): Controls the verbosity of the output during feature selection. `0` for minimal output, higher values for more detailed information. Defaults to `0`.

<h3>Old Method vs. New Method</h3>

**Select either the old featurewiz method or the new method** using the `classic` argument in the new library: (e.g., if you set `classic`=True, you will get features similar to the old feature selection method). If you set it to False, you will use the new feature selection method. I would suggest you try both methods to see which set of features works well for your dataset.<br>

![old_vs_new_method](./images/old_vs_new_featurewiz.png)

The new `featurewiz-polars` library uses an improved method for `recursive_xgboost` feature selection known as `Split-Driven Recursive_XGBoost`: In this method, we use Polars under the hood to speed up calculations for large datasets and in addition perform the following steps:
1.	**Split Data for Validation**: Divide the dataset into separate training and validation sets. The training set is used to build the XGBoost model, and the validation set is used to evaluate how well the selected features generalize to unseen data.
2.	**XGBoost Feature Ranking (with Validation)**: Within each run, use the training set to train an XGBoost model and evaluate feature importance. Assess the performance of selected features on the validation set to ensure they generalize well.
3.	**Select Key Features (with Validation)**: Determine the most significant features based on their importance scores and validation performance.
4.	**Repeat with New Split**: After each run of the recursive_xgboost cycle is complete, repeat the entire process (splitting, ranking, selecting) with a new train/validation split.
5.	**Final, Stabilized Feature Set**: After multiple runs with different splits, combine the selected features from all runs, removing duplicates. This results in a more stable and reliable final feature set, as it's less sensitive to the specific training/validation split used.

<h3>Benefits of using featurewiz-polars</h3>
<ul>
    <li><b>Significant Performance Boost:</b> Leverage Polars' speed and efficiency for feature engineering on large datasets.</li>
    <li><b>Native Polars Workflows:</b> Work directly with Polars DataFrames throughout your feature engineering and machine learning pipelines, avoiding unnecessary data conversions.</li>
    <li><b>Robust Feature Selection:</b> Benefit from the power of MRMR feature selection, optimized for Polars and corrected for accurate redundancy calculation across mixed data types.</li>
    <li><b>Flexible Categorical Encoding:</b> Choose from various encoding schemes (Target, WOE, Ordinal, OneHot Encoding)</li>
</ul>

<h3>Feedback and comments welcome</h3>

If you are working on processing massive datasets with Polars' speed and efficiency, while leveraging the power of `featurewiz_polars` for building high quality MLOps workflows, I welcome your feedback and comments to me at rsesha2001 at yahoo dot com for making it more useful to you in the months to come. Please `star` this repo or open a pull request or report an issue. Every which way, you make this repo more useful and better for everyone!

## License
Apache License 2.0

Copyright & All Rights Reserved 2025 Ram Seshadri