#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
from FeatureSelection.BaseFeatureSelection import BaseFeatureSelection as _BaseFeatureSelection
from sklearn.feature_selection import SelectKBest, chi2

class Chi2(_BaseFeatureSelection):
    def __init__(self, X_train, Y_train):
        super(Chi2, self).__init__(X_train, Y_train)

    def fit(self):
        feature_selector = SelectKBest(chi2, k="all")

        start_time = time.time()

        min_value = self.X_train.min()
        is_min_negative = min_value<0
        if is_min_negative.any():
            self.X_train[is_min_negative[is_min_negative].index] += min_value[is_min_negative].abs()

        feature_selector.fit(self.X_train, self.Y_train)
        self._fit_time = time.time() - start_time

        self._feature_score = pd.Series({self.X_train.columns[i]:feature_selector.scores_[i] for i in range(len(self.X_train.columns))})
















