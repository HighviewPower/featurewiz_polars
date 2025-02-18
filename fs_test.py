import polars as pl
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pdb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from featurewiz_polars import print_classification_metrics, print_regression_metrics
from featurewiz_polars import Featurewiz_MRMR, Featurewiz_MRMR_Pipeline
###########################################################################
n_features = 20
n_informative = 10
#X, y = make_classification(n_samples = 5000, n_features = n_features, 
#                n_informative = n_informative, n_redundant=4, random_state=42)
datapath = "../../../ram/Documents/Ram/data_sets/"
filename = "abalone.csv"
fl = pl.scan_csv(datapath+filename, null_values='NULL')
df = fl.collect().sample(5000)
print('Loaded data...', df.shape)
target = 'Rings' # Replace with your target column name
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
    import time
    import mrmr
    from mrmr import mrmr_classif
    start_time = time.time()
    #mrmr_feats = mrmr_classif(X=X_train, y=y_train, K=int(0.5*n_features))
    mrmr_feats = mrmr.polars.mrmr_regression(df=df, target_column=target, K=int(0.5*n_features))
    if model_type == 'Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[mrmr_feats], y_train)
    y_mrmr = model.predict(X_test[mrmr_feats])
    if model_type == 'Regression':
        print_regression_metrics(y_test.to_numpy().ravel(), y_mrmr, verbose=1)
    else:
        print_classification_metrics(y_test.to_numpy().ravel(), y_mrmr, verbose=1)
    print(f'{len(mrmr_feats)} features selected by mrmr: {mrmr_feats}')
    print('Time taken with MRMR = %0.1f seconds' %(
        time.time()-start_time))
    ##################################################################################
    from polars_sulov_mrmr import Sulov_MRMR
    start_time = time.time()
    mrmr = Featurewiz_MRMR_Pipeline(model_type=model_type, encoding_type='target')
    #mrmr = Sulov_MRMR(corr_threshold=0.7, verbose=1, n_recursions=5)
    mrmr.fit(X_train, y_train)
    pols_feats = mrmr.mrmr[-1].selected_features
    y_pols = mrmr.predict(X_test)
    if model_type == 'Regression':
        print_regression_metrics(y_test.to_numpy().ravel(), y_pols, verbose=1)
    else:
        print_classification_metrics(y_test.to_numpy().ravel(), y_pols, verbose=1)
    print('Time taken with PolarsFeaturewiz_MRMR = %0.1f seconds' %(time.time()-start_time))
    