<h1>featurewiz-polars</h1><h2>Blazing fast feature engineering and selection using mRMR and Polars</h2>

![featurewiz_polars_logo](./images/featurewiz_polars_taking_off.jpg)

<h3>Project Description</h3>

Supercharge your AI engineering pipelines with `featurewiz-polars`, a new library built on the classic `featurewiz` library, enhanced for high-performance feature engineering and selection using <b>Polars DataFrames</b>. 

<h3>Breaking News! featurewiz-polars is now available on pypi!</h3>

we are excited to announce that after extensive testing, featurewiz-polars is now available on pypi! You can now install it using pip:

```
pip install featurewiz-polars
```

This means that you can now easily install and use `featurewiz-polars` in your Python projects. We hope you find it useful and we welcome any feedback or contributions you may have.

<h3>Motivation: Addressing Bottlenecks</h3>

<p>This library was born out of the need for <b>efficient feature engineering</b> when working with <b>large datasets</b> and the <b>Polars DataFrame library</b>. Traditional feature selection and categorical encoding methods often become computationally expensive and memory-intensive as datasets grow in size and dimensionality.</p>

<p>Specifically, the motivation stemmed from the following challenges:</p>

<ol>
    <li><b>Performance limitations with large datasets:</b> Existing feature selection and encoding implementations (often in scikit-learn or Pandas-based libraries) can be slow and inefficient when applied to datasets with millions of rows and hundreds of columns.</li>          
    <li><b>Lack of Polars integration:</b> Many feature engineering tools are designed for Pandas DataFrames, requiring conversions to and from Polars, which introduces overhead and negates the performance benefits of Polars.</li>
    <li><b>Need for efficient MRMR feature selection:</b> Max-Relevance and Min-Redundancy (MRMR) is a powerful feature selection technique, but efficient implementations optimized for large datasets were needed, especially within the Polars ecosystem.</li>
    <li><b>Overfitting concerns:</b> Classic feature selection methods can lead to overfitting, especially when dealing with high-dimensional data. There was a need for a method that could provide more stable and reliable feature selection.</li>
    <li><b>Availability of scikit-learn Pipelines & Testing:</b> Most specialized libraries did not make their feature selection available as sckit-learn compatible transformers. That's why I created Pipeline examples (e.g., <code>fs_test.py</code> and <code>featurewiz_polars_test1.ipynb</code>) to  test the integration of our feature selector within scikit-learn pipelines to test the functionality and performance of the library.</li>
    <li><b>Addressing Date-Time Variables:</b> Most feature selection libraries did not handle date-time variables. Hence they were useless in time series problems. Recognizing this gap, the library's scope was expanded to include date-time variables (to help in time series tasks) and also the filling of NaN's and Nulls which commonly occurs in in Polars data frames. Thus a <code>Polars_MissingTransformer</code> was  added to handle Nulls and NaNs efficiently within Polars.</li>
    <li><b>Testing and Refinement:</b> Throughout the development, I put in an intense focus on verifying the correctness of the new algorithms and code, making sure that the new algorithm outpeformed my existing classic featurewiz library, particularly in the <code>recursive_xgboost</code> method which I modified.</li>
</ol>

<h3>Key Differentiators</h3>

The new `featurewiz-polars` leverages Polars library to deliver following **advantages over the current classic `featurewiz` library:**

1.  **Conquer Overfitting:** *`featurewiz-polars` validates features on a train-validation split, crushing overfitting and boosting generalization power.* 

2.  **Rock-Solid Stability:** *Multiple runs with different splits mean more stable feature selection. Thanks to Polars, stabilization is now lightning fast!* 

3.  **Big Data? No Sweat!** *Polars' raw speed and efficiency tame even the largest datasets, making feature selection a breeze.* 

4.  **XGBoost / Polars native integration:** *`featurewiz-polars` integrates natively and seamlessly with XGBoost, streamlining your entire ML pipeline from start to finish with Polars.*

In short, using Polars with our train-validation-split `recursive_xgboost` method offers a powerful combination: the robust feature selection of classic `featurewiz` with the new stabilization tailwind to give you more choices for reliable and fast feature selection, particularly for large datasets.

<h3> Install</h3>

The `featurewiz-polars` library is not available on pypi yet (I am still refining it). There are 3 ways to download and install this library to your machine.

1. You can git clone this library from source and run it from the terminal command as follows:

```
git clone https://github.com/AutoViML/featurewiz_polars.git
cd featurewiz_polars
pip install -r requirements.txt
cd examples
python fs_test.py
```
or<br>
2. You can download and unzip https://github.com/AutoViML/featurewiz_polars/archive/master.zip and follow the instructions from pip install above. But you start from the terminal in the directory where you downloaded the zip file.<br>
or<br>
3. You can install either from source or from pypi as follows on the terminal command:

```
pip install git+https://github.com/AutoViML/featurewiz_polars.git
or
pip install featurewiz-polars
```

<h3>Feeding data into featurewiz-polars</h3>

To help you quickly get started with the `featurewiz-polars` library, I've provided example scripts like `fs_test.py`. These scripts demonstrate how to use the library in a concise manner. Additionally, the `fs_lazytransform_test.py` script allows you to compare the performance of `featurewiz-polars` against the `lazytransform` library. For a more in-depth comparison, use `fs_mr_comparison_test.py` to benchmark `featurewiz-polars` against other competitive mRMR feature selection libraries. 

If you prefer working in a Jupyter Notebook or Colab, here are direct links to work in Colab with featurewiz-polars:

## Open In Colab Notebooks

Anybody can open a copy of my Github-hosted notebooks within Colab. To make it easier I have created `Open-in-Colab` links to those GitHub-hosted notebooks below:

<h4>Featurewiz-Polars Test Notebook</h4>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AutoViML/featurewiz_polars/blob/main/examples/featurewiz-polars-test.ipynb)

<h4>Featurewiz-Polars vs classic featurewiz comparison test</h4>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AutoViML/featurewiz_polars/blob/main/examples/fw_polars_vs_featurewiz_test.ipynb)

I have also provided code snippets to illustrate how to load a file into `polars` library's dataframes for use with `featurewiz-polars`.

<ul>   

    ### Load data into Polars DataFrames using:

    import polars as pl
    df = pl.read_csv(datapath+filename, null_values=['NULL','NA'], try_parse_dates=True,
        infer_schema_length=10000, ignore_errors=True)

    ### Before we do feature selection we always need to make sure we split the data #######
    target = 'target'
    predictors = [x for x in df.columns if x not in [target]]

    X = df[predictors]
    y = df[target] 

    ##############  BEWARE WHEN USING SKLEARN TRAIN_TEST_SPLIT WITH POLARS DATA #######################
    # If you perform train-test split using sklearn gives different random samples each time
    # So this doesn't work: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Instead you must split using polars_train_test_split with seed parameter to get same random samples 
    ####################################################################################################
    from featurewiz_polars import polars_train_test_split
    X_train, X_test, y_train, y_test = polars_train_test_split(X, y, test_size=0.2, random_state=42)

</ul>

## Feature Selection with `featurewiz-polars`: Two Approaches

### 1. Feature Selection Only with `FeatureWiz`

This approach is useful when you want to pre-process your data and select the most relevant features *before* feeding them into a separate model training pipeline.

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

### 2. Feature Selection and Model Training with `FeatureWiz_Model`

This approach combines feature selection and model training into a single pipeline.

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
```

**Key Points:**

*   We use the `FeatureWiz_Model` class to combine feature selection and model training.
*   The `fit_transform` method is used to fit the feature selection process *and* train the specified model on the training data.
*   The `predict` method handles both transforming the test data using the learned feature selection and making predictions with the trained model, streamlining the entire process.

<h3>Arguments for featurewiz_polars Pipeline</h3>

The `FeatureWiz_Model` class initializes the pipeline with a built-in Random Forest estimator (which you can change - see below) for building data pipelines that use the feature engineering, selection, and model training capabilities of Polars. You need to upload your data into Polars DataFrames and then start calling these pipelines.

#### Arguments:

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


Copyright 2025 Ram Seshadri