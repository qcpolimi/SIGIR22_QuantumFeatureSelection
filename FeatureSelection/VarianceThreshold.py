#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
from FeatureSelection.BaseFeatureSelection import BaseFeatureSelection as _BaseFeatureSelection
from sklearn.feature_selection import VarianceThreshold as VarianceThreshold_skopt

class VarianceThreshold(_BaseFeatureSelection):
    def __init__(self, X_train, Y_train):
        super(VarianceThreshold, self).__init__(X_train, Y_train)

    def fit(self):
        feature_selector = VarianceThreshold_skopt()
        start_time = time.time()
        feature_selector.fit(self.X_train, self.Y_train)
        self._fit_time = time.time() - start_time

        self._feature_score = pd.Series({self.X_train.columns[i]:feature_selector.variances_[i] for i in range(len(self.X_train.columns))})
















