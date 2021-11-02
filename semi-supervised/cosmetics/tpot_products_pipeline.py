import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Average CV score on the training set was: 0.7877905407895602
exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.8500000000000001, n_estimators=100), threshold=0.30000000000000004),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=6, max_features=0.45, min_samples_leaf=11, min_samples_split=14, n_estimators=100, subsample=0.7000000000000001)
)


