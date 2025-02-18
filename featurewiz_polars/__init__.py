# -*- coding: utf-8 -*-
################################################################################
#     featurewiz - advanced feature engineering and best features selection in single line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
from .featurewiz import featurewiz_polars
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""%s featurewiz_polars %s. Use the following syntax:
    >>> wiz = Polars_FeatureWiz_MRMR(verbose=0)
    >>> X_train_selected, y_train = wiz.fit_transform(X_train, y_train)
    >>> X_test_selected = wiz.transform(X_test)
    >>> selected_features = wiz.features
    """ %(module_type, version_number))
################################################################################
