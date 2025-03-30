# Featurewiz-Polars 🚀

[![PyPI version](https://img.shields.io/pypi/v/featurewiz_polars.svg)](https://pypi.org/project/featurewiz_polars/)
[![License: Apache2.0](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://opensource.org/licenses/Apache2.0)
[![Build Status](https://img.shields.io/github/actions/workflow/status/AutoViML/featurewiz_polars/ci.yml?branch=main)](https://github.com/AutoViML/featurewiz_polars/actions) <!-- Placeholder URL -->
[![Coverage Status](https://img.shields.io/codecov/c/github/AutoViML/featurewiz_polars/main.svg)](https://codecov.io/gh/AutoViML/featurewiz_polars) <!-- Placeholder URL -->

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

## 📖 Usage & Examples
For more detailed usage instructions, explanations of the parameters, and advanced examples, please refer to:

Examples Directory: ./examples/ <!-- Placeholder link: Create this directory -->

Check out the Jupyter notebooks and scripts here for practical use cases.

API Documentation: [Link to Hosted Docs] <!-- Placeholder link: e.g., ReadTheDocs URL -->

## 🤝 Contributing
Contributions are welcome! Whether it's bug reports, feature requests, or code contributions, please get involved.

Check the Issues tab for existing bugs or feature discussions.

Review the CONTRIBUTING.md file (link to be created) for guidelines on how to contribute.

Fork the repository, create your feature branch, and submit a pull request.

## 📜 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
