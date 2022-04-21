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

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None])+1e-9)


class LinearPearsonCorrelation(_BaseFeatureSelection):
    def __init__(self, X_train, Y_train):
        super(LinearPearsonCorrelation, self).__init__(X_train, Y_train)

    def fit(self):
        start_time = time.time()
        feature_correlation = corr2_coeff(self.X_train.to_numpy().T, np.expand_dims(self.Y_train, axis=1).T).ravel()
        self._fit_time = time.time() - start_time

        self._feature_score = pd.Series({self.X_train.columns[i]:feature_correlation[i] for i in range(len(self.X_train.columns))})
















