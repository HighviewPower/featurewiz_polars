# -*- coding: utf-8 -*-
################################################################################
# featurewiz-polars - Blazing Fast MRMR feature selection using Polars for large datasets
# Python v3.12+
# Created by Ram Seshadri
# Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz_polars import FeatureWiz, FeatureWiz_Model, Featurewiz_MRMR, Featurewiz_MRMR_Model
from .polars_other_transformers import Polars_ColumnEncoder, YTransformer
from .polars_other_transformers import Polars_MissingTransformer
from .polars_categorical_encoder import Polars_CategoricalEncoder
from .polars_datetime_transformer import Polars_DateTimeTransformer
from .polars_sulov_mrmr import Sulov_MRMR, polars_train_test_split
from .print_metrics import print_regression_metrics, print_classification_metrics
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s featurewiz_polars %s. Use the following syntax:
 >> from featurewiz_polars import FeatureWiz, FeatureWiz_Model
    """ %(module_type, version_number))
################################################################################
