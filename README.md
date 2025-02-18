<!DOCTYPE html>
<html>
<head>
<title>featurewiz-polars: Efficient Feature Selection and Encoding for Polars DataFrames</title>
</head>
<body>

<h1>featurewiz-polars: Efficient Feature Selection and Encoding for Polars DataFrames</h1>

<h2>Project Tagline</h2>

<p>Supercharge your data pipelines with <b>featurewiz-polars</b>, a library designed for high-performance feature selection and categorical encoding specifically optimized for <b>Polars DataFrames</b>. Process massive datasets with speed and efficiency, leveraging the power of Polars for feature engineering and machine learning workflows.</p>

<h2>Motivation: Addressing Bottlenecks in Feature Engineering with Large Datasets</h2>

<p>This library was born out of the need for <b>efficient feature engineering</b> when working with <b>large datasets</b> and the <b>Polars DataFrame library</b>. Traditional feature selection and categorical encoding methods often become computationally expensive and memory-intensive as datasets grow in size and dimensionality.</p>

<p>Specifically, the motivation stemmed from the following challenges:</p>

<ul>
    <li><b>Performance limitations with large datasets:</b> Existing feature selection and encoding implementations (often in scikit-learn or Pandas-based libraries) can be slow and inefficient when applied to datasets with millions of rows and hundreds of columns.</li>
    <li><b>Lack of Polars integration:</b> Many feature engineering tools are designed for Pandas DataFrames, requiring conversions to and from Polars, which introduces overhead and negates the performance benefits of Polars.</li>
    <li><b>Need for efficient MRMR feature selection:</b> Max-Relevance and Min-Redundancy (MRMR) is a powerful feature selection technique, but efficient implementations optimized for large datasets were needed, especially within the Polars ecosystem.</li>
    <li><b>Handling diverse data types:</b> The library needed to seamlessly handle both numerical and categorical features, and intelligently apply appropriate mutual information estimators for redundancy calculations.</li>
    <li><b>Extensibility and Pipeline Integration:</b> The components should be designed as scikit-learn compatible transformers, allowing for easy integration into machine learning pipelines and workflows.</li>
</ul>

<h2>Key Features of featurewiz-polars</h2>

<ul>
    <li><b>Polars DataFrame Native:</b> Designed from the ground up to work directly with Polars DataFrames, maximizing performance and minimizing data conversion overhead.</li>
    <li><b>Efficient Categorical Encoding:</b> <code>PolarsCategoricalFeatureEncoderV2</code> provides high-speed categorical encoding (Target Encoding, WOE Encoding, One-Hot Encoding) leveraging Polars' optimized operations.</li>
    <li><b>Polars-optimized MRMR Feature Selection:</b> <code>PolarsMRMRSelectorV5</code> implements the MRMR feature selection algorithm with significant performance improvements by utilizing Polars' parallel processing capabilities and efficient data manipulation.</li>
    <li><b>Corrected Mutual Information Estimation:</b> Intelligently selects the appropriate mutual information estimator (<code>mutual_info_classif</code> or <code>mutual_info_regression</code>) within the MRMR redundancy calculation based on the data types of the features being compared, ensuring accurate redundancy assessment for mixed data types.</li>
    <li><b>Scikit-learn Pipeline Compatible:</b> Built as scikit-learn compatible transformers, allowing seamless integration into existing scikit-learn pipelines and workflows.</li>
    <li><b>Post-Selection Refinement (Optional):</b> <code>PolarsMRMRSelectorV5</code> includes an optional post-selection refinement step using a classifier (RandomForest) to further optimize the selected feature subset based on performance on a validation set.</li>
    <li><b>Date-Time Feature Ready:</b> Designed to be easily extensible to include date-time feature transformers (e.g., <code>PolarsDateTimeFeatureTransformer</code>), recognizing the importance of temporal features in many datasets.</li>
</ul>

<h2>Development Steps: From Request to Algorithm</h2>

<p>The development of <code>featurewiz-polars</code> was an iterative process driven by the initial request for improved feature engineering and evolved through several key stages:</p>

<ol>
    <li><b>Initial Request & Problem Definition:</b> My journey began with a user request to bring featurewiz' amazing feature selection and encoding to Polars particularly for large datasets. The core problem was identified as the inefficiency of existing methods which cannot scale with pandas to millions of rows but can be done with Polars.</li>
    <li><b>Focus on Polars Efficiency:</b> The decision was made to build the library natively for Polars DataFrames to leverage Polars' speed and memory efficiency. This meant implementing my feature engineering algorithm using native Polars operations as much as possible.</li>
    <li><b><code>Polars_CategoricalEncoder</code> Development:</b> The first step was to create a fast categorical encoder for Polars. <code>Polars_CategoricalEncoder</code> was initially developed, followed by refining the implementation, by adding more encoding types (Target, WOE, Ordinal), and I then finally added handling of nan's and null values in those categories.</li>
    <li><b><code>Polars_DateTimeEncoder</code> Development:</b> The next step was to create a fast date-time encoder for Polars. <code>Polars_DateTimeEncoder</code> was developed, again added some error handling in dates.</li>
    <li><b><code>Other Transformers</code> Development:</b> The final step was to create a Y-Transformer that can encode target variables that are categorical for Polars. <code>The YTransformer</code> was developed, again adding some nifty error handling to incorporate scikit-learn pipelines.</li>
    <li><b><code>Polars_SULOV_MRMR</code> Development & Iteration:</b> The core of the library, the <code>Polars_SULOV_MRMR</code>, underwent several iterations to optimize performance and correctness:
        <ul>
            <li><b><code>V1-V3</code>:</b> Initial implementations focused on creating a basic Polars-compatible MRMR selector, leveraging my featurewiz implementation and modifying it to Polars for data manipulation and potentially <code>joblib</code> for parallelization (later removed instead for Polars' own internal parallelism).</li>
            <li><b><code>V4</code>:</b> A critical correction was made to the recursive XGBoost calculation in <code>recursive_xgboost</code> to address the growing volume of features selected where there was not enough signal.</li>
            <li><b><code>V5</code>:</b> A major stumbling block was identified and corrected in <code>V5</code>. This involved ensuring stable number of features selected in each round of XGBoost calculation, and ensuring almost the same best features are selected. This is still not a 100% consistent application yet ensures that correct information features are selected in the final round.</li>
        </ul>
    </li>
    <li><b>Pipeline Examples & Testing:</b> Pipeline examples (e.g., <code>fs_test.py</code> and <code>featurewiz_polars_test1.ipynb</code>) were created to test the integration of the encoder and selector within scikit-learn pipelines and to test the functionality and performance of the library.</li>
    <li><b>Addressing Regression and Date-Time Variables:</b> The library's scope was expanded to include regression tasks and the importance of filling NaN's and Nulls in Polars data. A <code>Polars_MissingTransformer</code> was also added to handle Nulls and NaNs in variables efficiently within Polars.</li>
    <li><b>Verification and Refinement:</b> Throughout the development, there was a focus on verifying the correctness of the algorithms and code, leading to the identification and correction of errors, particularly in the <code>recursive_xgboost</code> method.</li>
</ol>

<h2>Example Usage</h2>

<p>See the example pipeline scripts (e.g., <code>fs_test.py</code>) for demonstrations of how to use <code>featurewiz-polars</code> components in a machine learning pipeline. These examples showcase:</p>

<ul>
    <li>Loading data into Polars DataFrames.</li>
    <li>Using <code>Featurewiz_MRMR</code> for doing feature selection in scikit-learn pipelines.</li>
    <li>Employing an additional model to make <code>Featurewiz_MRMR_Model</code> for efficient testing of feature selection.</li>
    <li>Integrating these components with scikit-learn Pipelines (e.g., including a RandomForestClassifier as a defauolt model to test these pipelines).</li>
</ul>

<h2>Benefits of using featurewiz-polars</h2>

<ul>
    <li><b>Significant Performance Gains:</b> Leverage Polars' speed and efficiency for feature engineering on large datasets.</li>
    <li><b>Simplified Polars Workflows:</b> Work directly with Polars DataFrames throughout your feature engineering and machine learning pipelines, avoiding unnecessary data conversions.</li>
    <li><b>Robust Feature Selection:</b> Benefit from the power of MRMR feature selection, optimized for Polars and corrected for accurate redundancy calculation across mixed data types.</li>
    <li><b>Flexible Categorical Encoding:</b> Choose from various encoding schemes (Target, WOE, Ordinal Encoding)</li>
</ul>
