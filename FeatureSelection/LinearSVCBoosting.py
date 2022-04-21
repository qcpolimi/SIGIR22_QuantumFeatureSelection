#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
import numpy as np
from FeatureSelection.BaseFeatureSelection import BaseFeatureSelection as _BaseFeatureSelection
from FeatureSelection.BaseFeatureSelection import scale_data
from FeatureSelection.LinearPearsonCorrelation import corr2_coeff
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class LinearSVCBoosting(_BaseFeatureSelection):
    def __init__(self, X_train, Y_train):
        super(LinearSVCBoosting, self).__init__(X_train, Y_train)

    def fit(self):
        start_time = time.time()

        X_train_scaled = scale_data(self.X_train)
        X_train_predictions = pd.DataFrame().reindex_like(X_train_scaled)

        # The feature values are replaced by the predictions made by a SVClassifier that only uses that feature
        for feature_name in X_train_scaled.columns:
            selected_feature = X_train_scaled[feature_name].to_numpy().reshape(-1, 1)
            precictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(selected_feature, self.Y_train).predict(selected_feature)
            X_train_predictions[feature_name] = precictions

        feature_correlation = corr2_coeff(X_train_predictions.to_numpy().T, np.expand_dims(self.Y_train, axis=1).T).ravel()
        self._fit_time = time.time() - start_time

        self._feature_score = pd.Series({X_train_predictions.columns[i]:feature_correlation[i] for i in range(len(X_train_predictions.columns))})
















