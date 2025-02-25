import polars as pl
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from featurewiz_polars import print_classification_metrics, print_regression_metrics
from featurewiz_polars import Featurewiz_MRMR, Featurewiz_MRMR_Model
from featurewiz_polars import Sulov_MRMR
import time
import pdb
###########################################################################
n_features = 20
n_informative = 10
#X, y = make_classification(n_samples = 5000, n_features = n_features, 
#                n_informative = n_informative, n_redundant=4, random_state=42)
datapath = "../data/"
filename = "house_price_regression.csv"
fl = pl.scan_csv(datapath+filename, null_values=['NULL','NA'], try_parse_dates=True, infer_schema_length=10000, ignore_errors=True)
df = fl.collect()#.sample(5000)
print('Loaded data...', df.shape)
target = 'price' # Replace with your target column name
model_type = 'Regression'
if target not in df.columns:
    print(f"Error: Target column '{target}' not found in the CSV file.")
    exit()
predictors = [x for x in df.columns if x!=target]
X = df[predictors]
y = df[target]
print('Data dimensions (rows x cols) = %d dims' %(int(X.shape[0]*X.shape[1])))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# select top 10 features using mRMR
if __name__ == '__main__':
    from featurewiz import LazyTransformer, FeatureWiz
    start_time = time.time()
    lazy = LazyTransformer()
    X_l, y_l = lazy.fit_transform(X_train.to_pandas(), y_train.to_pandas())
    X_tl = lazy.transform(X_test.to_pandas())
    wiz = FeatureWiz(corr_limit=0.7)
    wiz.fit_transform(X_l, y_l)
    mrmr_feats = wiz.features
    if model_type == 'Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_l[mrmr_feats], y_l)
    y_mrmr = model.predict(X_tl[mrmr_feats])
    if model_type == 'Regression':
        print_regression_metrics(y_test.to_numpy().ravel(), y_mrmr, verbose=1)
    else:
        print_classification_metrics(y_test.to_numpy().ravel(), y_mrmr, verbose=1)
    print(f'{len(mrmr_feats)} features selected by featurewiz-classic: {mrmr_feats}')
    print('Time taken with MRMR = %0.1f seconds' %(
        time.time()-start_time))
    ##################################################################################
    start_time = time.time()
    mrmr = Featurewiz_MRMR_Model(model_type=model_type, 
            corr_threshold=0.7, encoding_type='onehot', classic=True, verbose=0)
    #mrmr = Sulov_MRMR(corr_threshold=0.7, verbose=1, n_recursions=5)
    mrmr.fit_predict(X_train, y_train)
    pols_feats = mrmr.selected_features
    print(f'{len(pols_feats)} features selected by featurewiz-polars: {pols_feats}')
    y_pols = mrmr.predict(X_test)
    if model_type == 'Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    ###### if y_test needs to be transformed use this ####
    if model_type == 'Regression':
        print_regression_metrics(y_test.to_numpy().ravel(), y_pols, verbose=1)
    else:
        print_classification_metrics(mrmr.y_encoder.transform(y_test).to_numpy().ravel(), y_pols, verbose=1)
    print('Time taken with PolarsFeaturewiz_MRMR = %0.1f seconds' %(time.time()-start_time))
    ###################################################################################
