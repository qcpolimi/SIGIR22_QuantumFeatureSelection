#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
import numpy as np
import dimod
from FeatureSelection.BaseQUBOFeatureSelection import BaseQUBOFeatureSelection as _BaseQUBOFeatureSelection
from FeatureSelection.BaseFeatureSelection import scale_data
from FeatureSelection.LinearPearsonCorrelation import corr2_coeff
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class QUBOSVCBoosting(_BaseQUBOFeatureSelection):
    """
    The goal is to select feature such that:
    - Each feature is replaced by the predictions made by a SVC Classifier using that single feature
    - The correlation of that feature with the target variable is maximized
    - The correlation between selected features is minimized
    """

    def __init__(self, X_train, Y_train):
        super(QUBOSVCBoosting, self).__init__(X_train, Y_train)

    def fit(self, reg_lambda = 0.5):
        start_time = time.time()

        X_train_scaled = scale_data(self.X_train)
        X_train_predictions = pd.DataFrame().reindex_like(X_train_scaled)

        # The feature values are replaced by the predictions made by a SVClassifier that only uses that feature
        for feature_name in X_train_scaled.columns:
            selected_feature = X_train_scaled[feature_name].to_numpy().reshape(-1, 1)
            precictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(selected_feature, self.Y_train).predict(selected_feature)
            X_train_predictions[feature_name] = precictions


        # Vectorized correlation between each feature
        Q = X_train_predictions.corr(method='pearson')
        Q = Q.to_numpy()

        # Correlation with each feature and the label
        n_samples = len(X_train_scaled)
        n_features = len(X_train_scaled.columns)
        diagonal_correlations = corr2_coeff(X_train_predictions.to_numpy().T, np.expand_dims(self.Y_train, axis=1).T)

        np.fill_diagonal(Q, n_samples/(n_features**2) + reg_lambda - 2 * diagonal_correlations)

        # Replace nan with 0.0 and scale
        Q = np.nan_to_num(Q, copy=True, nan=0.0)
        Q = Q / np.max(Q)

        self._Q = Q

        self._fit_time = time.time() - start_time
















