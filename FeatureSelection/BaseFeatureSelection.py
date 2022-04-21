#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
from utils.DataIO import DataIO
from sklearn.preprocessing import StandardScaler

def scale_data(X_data):
    scaler = StandardScaler()
    scaler.fit(X_data)
    scaled_array = scaler.transform(X_data)
    X_train_scaled = pd.DataFrame(scaled_array, columns=X_data.columns)

    return X_train_scaled

class BaseFeatureSelection(object):
    def __init__(self, X_train, Y_train):
        super(BaseFeatureSelection, self).__init__()
        self.X_train = X_train
        self.Y_train = Y_train

        self._fit_time = None
        self._feature_score = None

    def fit(self):
        pass

    def select_best_k(self, k_largest):
        start_time = time.time()
        top_k_list = list(self._feature_score.nlargest(k_largest).keys())
        select_best_k_time = time.time() - start_time
        return top_k_list, select_best_k_time

    def save_model(self, folder_path, file_name):

        data_dict_to_save = {"_feature_score": self._feature_score,
                             "_fit_time": self._fit_time,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


    def load_model(self, folder_path, file_name):
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])
















