#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import itertools
import time

import numpy as np

from FeatureSelection.BaseQUBOFeatureSelection import BaseQUBOFeatureSelection as _BaseQUBOFeatureSelection
from PyMIToolbox import discAndCalcMutualInformation, discAndCalcConditionalMutualInformation


class QUBOMutualInformation(_BaseQUBOFeatureSelection):
    """
    The goal is to select feature such that:
    - The mutual information of that feature with the target variable is maximized
    - The mutual information of that feature with the target variable given other selected features is maximized

    The mutual information is computed for discrete variables using the MIToolbox library.
    Continuous variables are first discretized by taking their integer part: value is x is considered as floor(x).
    """

    def __init__(self, X_train, Y_train):
        super(QUBOMutualInformation, self).__init__(X_train, Y_train)

    def fit(self):
        start_time = time.time()

        features = self.X_train.columns
        n_features = len(features)
        n_samples = len(self.X_train)

        # Compute MI between features and target
        mi_iter = (discAndCalcMutualInformation(self.X_train[feature], self.Y_train) for feature in features)
        mi = np.fromiter(mi_iter, dtype=np.double, count=n_features)

        # Compute conditional MI between features and target given other features
        cmi_iter = (discAndCalcConditionalMutualInformation(self.X_train[f1], self.Y_train, self.X_train[f2])
                    for f1, f2 in itertools.permutations(features, 2))
        cmi = np.fromiter(cmi_iter, dtype=np.double, count=n_features ** 2 - n_features)

        # Build matrix Q from computed arrays
        Q = np.zeros((n_features, n_features))
        Q[np.logical_not(np.eye(n_features))] = cmi

        # # Symmetric to upper triangular
        # Q = np.triu(Q) + np.triu(Q.T)

        # Fill the diagonal with MI between features and target
        np.fill_diagonal(Q, mi)

        # Replace nan with 0.0 and scale
        Q = np.nan_to_num(Q, copy=True, nan=0.0)
        Q = Q / np.max(Q)

        self._Q = -Q

        self._fit_time = time.time() - start_time
