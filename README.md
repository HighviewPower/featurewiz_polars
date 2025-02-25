<h1>featurewiz-polars</h1><h2>Blazing fast feature engineering and selection using mRMR and Polars</h2>

<h3>Project Description</h3>

Supercharge your AI engineering pipelines with `featurewiz-polars`, a new library built on the classic `featurewiz` library, enhanced for high-performance feature engineering and selection using <b>Polars DataFrames</b>. 

<h3>Motivation: Addressing Bottlenecks</h3>

<p>This library was born out of the need for <b>efficient feature engineering</b> when working with <b>large datasets</b> and the <b>Polars DataFrame library</b>. Traditional feature selection and categorical encoding methods often become computationally expensive and memory-intensive as datasets grow in size and dimensionality.</p>

<p>Specifically, the motivation stemmed from the following challenges:</p>

<ul>
    <li><b>Performance limitations with large datasets:</b> Existing feature selection and encoding implementations (often in scikit-learn or Pandas-based libraries) can be slow and inefficient when applied to datasets with millions of rows and hundreds of columns.</li>
    <li><b>Lack of Polars integration:</b> Many feature engineering tools are designed for Pandas DataFrames, requiring conversions to and from Polars, which introduces overhead and negates the performance benefits of Polars.</li>
    <li><b>Need for efficient MRMR feature selection:</b> Max-Relevance and Min-Redundancy (MRMR) is a powerful feature selection technique, but efficient implementations optimized for large datasets were needed, especially within the Polars ecosystem.</li>
    <li><b>Handling diverse data types:</b> The library needed to seamlessly handle both numerical and categorical features, and intelligently apply appropriate imputation strategies for diverse use cases.</li>
    <li><b>Extensibility and Pipeline Integration:</b> The components should be designed as scikit-learn compatible transformers, allowing for easy integration into machine learning pipelines and workflows.</li>
</ul>

<h3>Key Differentiators</h3>

The new `featurewiz-polars` leverages Polars library to deliver following **advantages over the current classic `featurewiz` library:**

1.  **Conquer Overfitting:** *`featurewiz-polars` validates features on a train-validation split, crushing overfitting and boosting generalization power.* 

2.  **Rock-Solid Stability:** *Multiple runs with different splits mean more stable feature selection. Thanks to Polars, stabilization is now lightning fast!* 

3.  **Big Data? No Sweat!** *Polars' raw speed and efficiency tame even the largest datasets, making feature selection a breeze.* 

4.  **XGBoost / Polars native integration:** *`featurewiz-polars` integrates natively and seamlessly with XGBoost, streamlining your entire ML pipeline from start to finish with Polars.*

In short, using Polars with our train-validation-split `recursive_xgboost` method offers a powerful combination: the robust feature selection of classic `featurewiz` with the new stabilization tailwind to give you more choices for reliable and fast feature selection, particularly for large datasets.

<h3>Development stages: From request to algorithm</h3>

The development of `featurewiz-polars`  was an iterative process driven by the initial request for improved feature engineering and evolved through several key stages:

<ol>
    <li><b>Initial Request & Problem Definition:</b> My journey began with a user request to bring featurewiz's amazing feature selection and encoding capabilities to Polars, particularly for large datasets. The core problem they faced was the inefficiency of existing methods which cannot scale with pandas to millions of rows.</li>
    <li><b>Focus on Polars Efficiency:</b> I then made the decision to build the library natively using Polars DataFrames to leverage Polars' blazing speed and memory efficiency. This meant re-implementing my entire feature engineering pipeline using native Polars operations as much as possible. This was much harder than I imagined.</li>
    <li><b><code>Polars_CategoricalEncoder</code> Development:</b> The first step was to create a fast categorical encoder for Polars. <code>Polars_CategoricalEncoder</code> was conceived and developed, by refining my original featurewiz implementation called <code>My_Label_Encoder()</code>, but adding more encoding types (Target, WOE, Ordinal, OneHot), and I then finally added handling of nan's and null values in those categories (Wooh).</li>
    <li><b><code>Polars_DateTimeEncoder</code> Development:</b> The next step was to create a fast date-time encoder for Polars. <code>Polars_DateTimeEncoder</code> was developed, again added some error handling in dates.</li>
    <li><b><code>Other Transformers</code> Development:</b> The final step was to create a Y-Transformer that can encode target variables that are categorical for Polars. <code>The YTransformer</code> was developed and it turned out to be very useful in scikit-learn pipelines which I later developed (see below) .</li>
    <li><b><code>Polars_SULOV_MRMR</code> Development & Iteration:</b> The core of the library, the <code>Polars_SULOV_MRMR</code>, underwent several iterations to optimize performance and correctness:
        <ul>
            <li><b><code>V1-V3</code>:</b> Initially I focused on creating a basic Polars-compatible MRMR selector, copying my featurewiz implementation and modifying it to Polars.</li>
            <li><b><code>V4</code>:</b> I then made a critical correction to the calculations in <code>recursive_xgboost</code> process to address the growing volume of features it selected that did not add to performance.</li>
            <li><b><code>V5</code>:</b> I found a major stumbling block in that classic process and corrected it in <code>V5</code>. This involved splitting the dataset into train and validation and running recursive_xgboost within it, thus  ensuring a stable number of features in each round of XGBoost feature selection. I have renamed the old approach as "classic" and the new approach as "split-driven" which you can test in the library.</li>
        </ul>
    </li>
    <li><b>Pipeline Examples & Testing:</b> Pipeline examples (e.g., <code>fs_test.py</code> and <code>featurewiz_polars_test1.ipynb</code>) were created to test the integration of the encoder and selector within scikit-learn pipelines and to test the functionality and performance of the library.</li>
    <li><b>Addressing Date-Time Variables:</b> The library's scope was expanded to include date-time variables (to help in time series tasks) and the filling of NaN's and Nulls in Polars data frames. Thus a <code>Polars_MissingTransformer</code> was  added to handle Nulls and NaNs efficiently within Polars.</li>
    <li><b>Testing and Refinement:</b> Throughout the development, I put in an intense focus on verifying the correctness of the new algorithms and code, making sure that the new algorithm outpeformed my existing classic featurewiz library, particularly in the <code>recursive_xgboost</code> method which I modified.</li>
</ol>

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
3. You can install from source as follows on the terminal command:

```
pip install git+https://github.com/AutoViML/featurewiz_polars.git
```

<h3>Feeding data into featurewiz-polars</h3>

To help you quickly get started with the `featurewiz-polars` library, I've provided example scripts like `fs_test.py`. These scripts demonstrate how to use the library in a concise manner. Additionally, the `fs_comparison_test.py` script allows you to compare the performance of `featurewiz-polars` against the classic `featurewiz` library. For a more in-depth comparison, use `fs_mr_comparison_test.py` to benchmark `featurewiz-polars` against other competitive mRMR feature selection libraries. 

If you prefer working in a Jupyter Notebook, here is a code snippet to illustrate how to load a file into `polars` library's dataframes for use with `featurewiz-polars`.

<ul>   

    ### Load data into Polars DataFrames using:

    import polars as pl
    df = pl.read_csv(datapath+filename, null_values=['NULL','NA'], try_parse_dates=True,
        infer_schema_length=10000, ignore_errors=True)

    ### Split the Polars dataframe into train and test using scikit-learn's train_test_split using:

    target = 'target'
    predictors = [x for x in df.columns if x not in [target]]
    from sklearn.model_selection import train_test_split
    X = df[predictors]
    y = df[target] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</ul>

## Feature Selection with `featurewiz-polars`: Two Approaches

This section demonstrates two distinct ways to use the `featurewiz-polars` library: `feature selection only`, and `feature selection combined with model training`.

### 1. Feature Selection Only with `Featurewiz_MRMR`

This approach is useful when you want to pre-process your data and select the most relevant features *before* feeding them into a separate model training pipeline. It's particularly helpful if you want to experiment with different models using the same selected features.

Here's how to use `Featurewiz_MRMR` for feature selection in a classification scenario, where the categorical target variable "y" needs to be transformed into a numerical representation:

```python
from featurewiz_polars import Featurewiz_MRMR

# Initialize Featurewiz_MRMR for classification with specified parameters
mrmr = Featurewiz_MRMR(model_type="Classification",
            corr_threshold=0.7, encoding_type='onehot', classic=True, verbose=0)

# Fit and transform the training data (X_train, y_train)
X_transformed, y_transformed = mrmr.fit_transform(X_train, y_train)

# Transform the test data (X_test)
X_test_transformed = mrmr.transform(X_test)

# Transform the test target variable (y_test)
y_test_transformed = mrmr.y_encoder.transform(y_test)
```

**Key Points:**

*   We use the `Featurewiz_MRMR` class for feature selection.
*   The `fit_transform` method is used to fit the feature selection process on the training data and simultaneously transform it.
*   We use the `transform` method separately to transform the test data, applying the same feature selection learned from the training data.
*   The `y_encoder` is used to transform the target variable if it's categorical.

### 2. Feature Selection and Model Training with `Featurewiz_MRMR_Model`

This approach combines feature selection and model training into a single pipeline. It's useful when you want to streamline the entire process and train a model directly on the selected features.

Here's how to use `Featurewiz_MRMR_Model` for both feature selection and model training in a regression scenario. It assumes that the target variable "y" is already numerical.

```python
from featurewiz_polars import Featurewiz_MRMR_Model
from xgboost import XGBRegressor

# Initialize Featurewiz_MRMR_Model for regression with an XGBoost Regressor
mrmr_model = Featurewiz_MRMR_Model(model_type="Regression", model=XGBRegressor(),
            corr_threshold=0.7, encoding_type='onehot', classic=True, verbose=0)

# Fit and transform the training data (X_train, y_train)
X_transformed, y_transformed = mrmr_model.fit_transform(X_train, y_train)

# Make predictions on the test data (X_test) - handles both transform and predict simultaneously
y_pred = mrmr_model.predict(X_test)
```

**Key Points:**

*   We use the `Featurewiz_MRMR_Model` class to combine feature selection and model training.
*   The `fit_transform` method is used to fit the feature selection process *and* train the specified model on the training data.
*   The `predict` method handles both transforming the test data using the learned feature selection and making predictions with the trained model, streamlining the entire process.

<h3>Arguments for featurewiz_polars Pipeline</h3>

The `Featurewiz_MRMR_Model` class initializes the pipeline with a built-in Random Forest estimator (which you can change - see below) for building data pipelines that use the feature engineering, selection, and model training capabilities of Polars. You need to upload your data into Polars DataFrames and then start calling these pipelines.

#### Arguments:

*   **`model`** (estimator object, *optional*): Any machine learning estimator can be passed in to be trained after feature selection. If `None`, a default estimator (Random Forest) will be used. Defaults to `None`.

*   **`model_type`** (str, *optional*): The type of model to be built (`'classification'` or `'regression'`). Determines the appropriate preprocessing and feature selection strategies. Defaults to `'classification'`.

*   **`encoding_type`** (str, *optional*): The type of encoding to apply to categorical features (`'target'`, `'onehot'`, etc.).  `'woe'` encoding is only available for classification model types. Defaults to `'target'`.

*   **`imputation_strategy`** (str, *optional*): The strategy for handling missing values (`'mean'`, `'median'`, `'zeros'`). Determines how missing data will be filled in before feature selection. Defaults to `'mean'`.

*   **`corr_threshold`** (float, *optional*): The correlation threshold for removing highly correlated features. Features with a correlation above this threshold will be targeted for removal. Defaults to `0.7`.

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