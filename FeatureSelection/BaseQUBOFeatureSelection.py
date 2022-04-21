#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""

import time
import dimod
import pandas as pd
import numpy as np
import neal
from utils.DataIO import DataIO
from FeatureSelection.BaseFeatureSelection import BaseFeatureSelection
from dwave.system.samplers import LeapHybridSampler
from dwave.system.composites import EmbeddingComposite



def json_convert_not_serializable(o):
    """
    Json cannot serialize automatically some data types, for example numpy integers (int32).
    This may be a limitation of numpy-json interfaces for Python 3.6 and may not occur in Python 3.7
    :param o:
    :return:
    """

    if isinstance(o, np.integer):
        return int(o)

    if isinstance(o, np.bool_):
        return bool(o)

    return o

def sample_wrapper(BQM, sampler, **sampler_hyperparams):

    if isinstance(sampler, EmbeddingComposite):
        sampler_hyperparams["return_embedding"] = True

        # Hyperparameters of type int32, int64, bool_ are not json serializable
        # transform them from numpy to native Python types
        for key,value in sampler_hyperparams.items():
            sampler_hyperparams[key] = json_convert_not_serializable(value)

    sampleset = sampler.sample(BQM, **sampler_hyperparams)
    sampleset_df = sampleset.aggregate().to_pandas_dataframe()
    sampler_info = sampleset.info

    # Change the format of the infos returned by the QPU sampler
    if isinstance(sampler, EmbeddingComposite):

        for key,value in sampler_info["timing"].items():
            sampler_info["timing_" + key] = value

        del sampler_info["timing"]

        for key,value in sampler_info["embedding_context"].items():
            sampler_info[key] = value

        del sampler_info["embedding_context"]

        for key,value in sampler_info.items():
            if "embedding" in key:
                sampler_info[key] = str(value)
                
    elif isinstance(sampler, LeapHybridSampler):
        pass
            
    elif isinstance(sampler, neal.SimulatedAnnealingSampler):
        sampler_info["beta_range_min"] = sampler_info["beta_range"][0]
        sampler_info["beta_range_max"] = sampler_info["beta_range"][1]
        del sampler_info["beta_range"]

    return sampleset_df, sampler_info





def _get_BQM_alpha(Q, alpha):

    diagonal = Q.diagonal()
    quadratic =  Q.copy()
    np.fill_diagonal(quadratic, 0)

    if alpha == "balanced":
        alpha = quadratic.mean() / (quadratic.mean() + diagonal.mean())

    Q = (1-alpha)*quadratic + alpha*diagonal

    # Using dimod.as_bqm(Q, "BINARY") creates an AdjVectorBQM which is not serializable
    BQM = dimod.binary_quadratic_model.BinaryQuadraticModel(dimod.as_bqm(Q, "BINARY"))

    return BQM, alpha





class BaseQUBOFeatureSelection(BaseFeatureSelection):
    def __init__(self, X_train, Y_train):
        super(BaseQUBOFeatureSelection, self).__init__(X_train, Y_train)

        self._Q = None


    def select_best_k(self, k_largest, solver, alpha,
                      sampler_hyperparams):
        """

        BQM = (1-alpha)*quadratic + alpha*diagonal
        """

        start_time = time.time()

        BQM, alpha = _get_BQM_alpha(self._Q, alpha)

        # Add k-combinations constraint and solve QUBO
        BQM_k = dimod.generators.combinations(BQM.num_variables, k_largest)
        BQM_k.update(BQM)

        sampleset_df, sampler_info = sample_wrapper(BQM_k, solver, **sampler_hyperparams)

        best_sample_index = sampleset_df["energy"].idxmin()
        best_sample_data = sampleset_df.loc[best_sample_index]

        top_k_list = [self.X_train.columns[i] for i in range(len(self.X_train.columns)) if  best_sample_data[i] == 1.0]

        select_best_k_time = time.time() - start_time

        return top_k_list, sampleset_df, sampler_info, select_best_k_time


    def save_model(self, folder_path, file_name):

        data_dict_to_save = {"_Q": self._Q,
                             "_fit_time": self._fit_time,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)













