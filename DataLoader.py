#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/12/2021

@author: Maurizio Ferrari Dacrema
"""

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.DataIO import DataIO
import pandas as pd

class DataLoader():

    def __init__(self, folder_path, dataset_name):

        dataIO = DataIO(folder_path + "data/")

        try:
            data_split = dataIO.load_data("data_split")

            for attrib_name, attrib_object in data_split.items():
                 self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            print("Dataset '{}' not found, downloading... ".format(dataset_name))

            # load dataset
            if dataset_name in ["tecator", "USPS", "gisette", "covertype", "20_newsgroups.drift"]:
                dataset_version = 2
            elif dataset_name in ["emotions"]:
                dataset_version = 3
            else:
                dataset_version = 1
            
            X_feature_all, Y_target_original = datasets.fetch_openml(name=dataset_name, version=dataset_version, return_X_y=True)

            if not isinstance(X_feature_all, pd.DataFrame):
                # Some feature matrices are sps sparse
                X_feature_all = pd.DataFrame(X_feature_all.toarray())

            label_encoder = LabelEncoder()
            label_encoder.fit(Y_target_original)
            Y_target_transform = label_encoder.transform(Y_target_original)

            ## FIX CATEGORICAL VARIABLES
            cat_columns = X_feature_all.select_dtypes(include=['category']).columns
            X_feature_all[cat_columns] = X_feature_all[cat_columns].apply(lambda x: x.cat.codes)

            cat_columns_1 = X_feature_all.select_dtypes(include=['object']).columns
            X_feature_all[cat_columns_1] = X_feature_all[cat_columns_1].apply(lambda x: x.to_numeric())

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_feature_all, Y_target_transform, test_size=0.3, stratify=Y_target_transform)

            data_split = {
                "X_train": self.X_train,
                "X_test": self.X_test,
                "Y_train": self.Y_train,
                "Y_test": self.Y_test,
                "class_mapper": {class_label:index for index,class_label in enumerate(label_encoder.classes_)}
            }

            dataIO.save_data("data_split", data_split)

            print("Dataset '{}' not found, downloading... Done!".format(dataset_name))
