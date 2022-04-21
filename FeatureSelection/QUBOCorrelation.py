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

class QUBOCorrelation(_BaseQUBOFeatureSelection):
    """
    The goal is to select feature such that:
    - The correlation of that feature with the target variable is maximized
    - The correlation between selected features is minimized
    """

    def __init__(self, X_train, Y_train):
        super(QUBOCorrelation, self).__init__(X_train, Y_train)

    def fit(self):
        start_time = time.time()

        X_train_scaled = scale_data(self.X_train)

        # Vectorized correlation between each feature
        Q = -X_train_scaled.corr(method='pearson')
        Q = Q.to_numpy()

        # Correlation with each feature and the label
        np.fill_diagonal(Q, corr2_coeff(X_train_scaled.to_numpy().T, np.expand_dims(self.Y_train, axis=1).T))

        # Replace nan with 0.0 and scale
        Q = np.nan_to_num(Q, copy=True, nan=0.0)
        Q = Q / np.max(Q)

        self._Q = -Q

        self._fit_time = time.time() - start_time
















