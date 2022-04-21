#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time

import pandas as pd

from FeatureSelection.BaseFeatureSelection import BaseFeatureSelection as _BaseFeatureSelection
from PyMIToolbox import discAndCalcMutualInformation


class LinearMutualInformationMIT(_BaseFeatureSelection):
    """
    The goal is to select feature such that the mutual information of that feature with the target variable is
    maximized.

    The mutual information is computed for discrete variables using the MIToolbox library.
    Continuous variables are first discretized by taking their integer part: value x is considered as floor(x).
    """

    def __init__(self, X_train, Y_train):
        super(LinearMutualInformationMIT, self).__init__(X_train, Y_train)

    def fit(self):
        start_time = time.time()
        self._feature_score = pd.Series(
            {feature: discAndCalcMutualInformation(self.X_train[feature], self.Y_train)
             for feature in self.X_train.columns})
        self._fit_time = time.time() - start_time
