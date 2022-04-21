#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/12/2021

@author: Maurizio Ferrari Dacrema
"""
from minorminer import busclique

import pandas as pd
from neal import SimulatedAnnealingSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import greedy, os, time, traceback, multiprocessing
import numpy as np
from utils.DataIO import DataIO
from tabu import TabuSampler
from DataLoader import DataLoader
from functools import partial

import minorminer, dimod
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler, LeapHybridSampler
from dwave.system import FixedEmbeddingComposite
import networkx as nx

from FeatureSelection.VarianceThreshold import VarianceThreshold
from FeatureSelection.Chi2 import Chi2
from FeatureSelection.ANOVA_F_Test import ANOVA_F_Test
# from FeatureSelection.LinearMutualInformation import LinearMutualInformation
from FeatureSelection.LinearMutualInformationMIT import LinearMutualInformationMIT
from FeatureSelection.LinearPearsonCorrelation import LinearPearsonCorrelation
from FeatureSelection.LinearSVCBoosting import LinearSVCBoosting

from FeatureSelection.QUBOCorrelation import QUBOCorrelation
from FeatureSelection.QUBOSVCBoosting import QUBOSVCBoosting
from FeatureSelection.QUBOMutualInformation import QUBOMutualInformation
from FeatureSelection.BaseQUBOFeatureSelection import _get_BQM_alpha

from utils.print_table_utils import print_global_result_summary, print_dataset_result_summary


def _k_comb_list(N, max_cases = 50):
    """
    List contains a maximum of 50 elements in [1, N)
    """

    if N <= max_cases:
        return np.arange(1, N)

    # Return all elements except the last one, which is N
    k_list = np.linspace(1, N, max_cases, dtype=int)
    k_list = np.unique(k_list)

    return k_list[:-1]



def _evaluate_classifier(X_train, Y_train, X_test, Y_test, classifier_folder, target_feature_k):

    classifier_instance = RandomForestClassifier()

    start_time = time.time()
    classifier_instance.fit(X_train, Y_train)
    classifier_algorithm_fit_time = time.time()-start_time

    Y_pred = classifier_instance.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)

    # If directory does not exist, create
    if not os.path.exists(classifier_folder):
        os.makedirs(classifier_folder)

    # dump(classifier_instance, classifier_folder + '/RandomForestClassifier_k_{}.joblib'.format(target_feature_k))

    # Compute and save the confusion matrix (TRAINING).
    # classifier_instance.confusion_train_matrix = confusion_matrix(Y_train, Y_pred)

    # Compute k-cross-validation scores and save them.
    start_time = time.time()
    CV_scores = cross_val_score(classifier_instance, X_train, Y_train, n_jobs=1, scoring='accuracy')
    classifier_algorithm_CV_time = time.time()-start_time

    return CV_scores, test_accuracy, classifier_algorithm_fit_time, classifier_algorithm_CV_time



def _all_features_dataset_experiment(result_dataset_folder, X_train, Y_train, X_test, Y_test):

    try:
        all_features_result_df = pd.read_csv(result_dataset_folder + "all_features_result_df.csv", index_col=0)
    except FileNotFoundError:
        all_features_result_df = pd.DataFrame(columns=["selection_algorithm_name",
                                        "actual_feature_k",
                                        "classifier_algorithm_name",
                                        "classifier_algorithm_fit_time",
                                        "classifier_algorithm_CV_time",
                                        "CV_scores",
                                        "CV_scores_mean",
                                        "CV_scores_std",])

        classifier_folder = result_dataset_folder + "all_features/"
        CV_scores, test_accuracy, classifier_algorithm_fit_time, classifier_algorithm_CV_time = _evaluate_classifier(X_train, Y_train, X_test, Y_test, classifier_folder, "all_features")

        all_features_result_df = all_features_result_df.append({
            "selection_algorithm_name": "all_features",
            "actual_feature_k": len(X_train.columns),
            "classifier_algorithm_name": "RandomForestClassifier",
            "classifier_algorithm_fit_time": classifier_algorithm_fit_time,
            "classifier_algorithm_CV_time": classifier_algorithm_CV_time,
            "CV_scores": CV_scores,
            "CV_scores_mean": CV_scores.mean(),
            "CV_scores_std": CV_scores.std(),
            "test_accuracy": test_accuracy,
        }, ignore_index=True)

        all_features_result_df.to_csv(result_dataset_folder + "all_features_result_df.csv", index=True)




def _generate_embedding(n_features, sampler):

    # BQM = dimod.binary_quadratic_model.BinaryQuadraticModel(dimod.as_bqm(Q, "BINARY"))

    start_time = time.time()

    # problem_graph = dimod.to_networkx_graph(BQM)
    # embedding = busclique.find_clique_embedding(n_features, topology)

    problem_graph = nx.complete_graph(n_features).edges()
    topology = sampler.to_networkx_graph()
    embedding = minorminer.find_embedding(problem_graph, topology)

    return embedding, time.time() -start_time


def _get_fixed_embedding(QUBO_solver, n_features, classifier_folder):

    dataIO = DataIO(classifier_folder)

    try:
        data_dict = dataIO.load_data("fixed_embedding")

        fixed_embedding = data_dict["fixed_embedding"]
        fixed_embedding = {int(key):value for key,value in fixed_embedding.items()}
        embedding_time = data_dict["embedding_time"]
    except FileNotFoundError:
        fixed_embedding, embedding_time = _generate_embedding(n_features, QUBO_solver)

        data_dict = {
            "fixed_embedding": fixed_embedding,
            "embedding_time": embedding_time,
        }

        dataIO.save_data("fixed_embedding", data_dict_to_save=data_dict)

    return fixed_embedding, embedding_time


def run_dataset_experiment(dataset_name, result_dataset_folder, classic_algorithms_dict, QUBO_algorithms_dict, QUBO_solvers_dict):

    data_loader = DataLoader(folder_path = result_dataset_folder, dataset_name = dataset_name)

    X_train = data_loader.X_train
    Y_train = data_loader.Y_train

    X_test = data_loader.X_test
    Y_test = data_loader.Y_test

    n_features = len(X_train.columns)

    print("Dataset: {}, Number of features: {}, Number of samples: {}".format(dataset_name, n_features, len(X_train)))

    k_comb_list = _k_comb_list(n_features, max_cases = 50)

    _all_features_dataset_experiment(result_dataset_folder, X_train, Y_train, X_test, Y_test)

    try:
        classic_result_df = pd.read_csv(result_dataset_folder + "classic_result_df.csv", index_col=0)
    except FileNotFoundError:
        classic_result_df = pd.DataFrame(columns=["selection_algorithm_name",
                                        "selection_algorithm_fit_time",
                                        "selection_algorithm_select_best_k_time",
                                        "selected_features",
                                        "target_feature_k",
                                        "actual_feature_k",
                                        "classifier_algorithm_name",
                                        "classifier_algorithm_fit_time",
                                        "classifier_algorithm_CV_time",
                                        "CV_scores",
                                        "CV_scores_mean",
                                        "CV_scores_std",
                                        "test_accuracy"])

    try:
        QUBO_result_df = pd.read_csv(result_dataset_folder + "QUBO_result_df.csv", index_col=0)
    except FileNotFoundError:
        QUBO_result_df = pd.DataFrame(columns=[*classic_result_df.columns,
                                        "QUBO_solver",
                                        "alpha_value",
                                        "alpha_heuristic",
                                        "embedding_time",])


    all_selection_algorithm_dict = {**classic_algorithms_dict, **QUBO_algorithms_dict}

    for selection_algorithm_name, selection_algorithm_class in all_selection_algorithm_dict.items():

        if dataset_name == "steel-plates-fault" and selection_algorithm_name in ["Linear Mutual Information MIToolbox", "QUBOMutualInformation"]:
            # For some reason it causes a segmentation fault
            continue

        classifier_folder = result_dataset_folder + selection_algorithm_name + "/"
        selection_algorithm_instance = selection_algorithm_class(X_train, Y_train)

        try:
            selection_algorithm_instance.load_model(classifier_folder, file_name="selection_algorithm_instance")
        except FileNotFoundError:
            print("Selection algorithm: {}, fitting...".format(selection_algorithm_name))
            selection_algorithm_instance.fit()
            selection_algorithm_instance.save_model(classifier_folder, file_name="selection_algorithm_instance")

        for target_feature_k in k_comb_list:

            if selection_algorithm_name in classic_algorithms_dict:

                # Check if it was already done
                if classic_result_df[(classic_result_df["selection_algorithm_name"]==selection_algorithm_name) &
                                     (classic_result_df["target_feature_k"]==target_feature_k)].empty:

                    print("Selection algorithm: {}, target number of features {}/{}".format(selection_algorithm_name, target_feature_k, n_features))

                    try:
                        selected_features, select_best_k_time = selection_algorithm_instance.select_best_k(target_feature_k)

                        X_train_selected_features = X_train[selected_features].copy()
                        X_test_selected_features = X_test[selected_features].copy()

                        CV_scores, test_accuracy, classifier_algorithm_fit_time, classifier_algorithm_CV_time = _evaluate_classifier(X_train_selected_features, Y_train, X_test_selected_features, Y_test, classifier_folder, target_feature_k)

                        classic_result_df = classic_result_df.append({
                            "selection_algorithm_name": selection_algorithm_name,
                            "selection_algorithm_fit_time": selection_algorithm_instance._fit_time,
                            "selection_algorithm_select_best_k_time": select_best_k_time,
                            "selected_features": selected_features,
                            "target_feature_k": target_feature_k,
                            "actual_feature_k": len(selected_features),
                            "classifier_algorithm_name": "RandomForestClassifier",
                            "classifier_algorithm_fit_time": classifier_algorithm_fit_time,
                            "classifier_algorithm_CV_time": classifier_algorithm_CV_time,
                            "CV_scores": CV_scores,
                            "CV_scores_mean": CV_scores.mean(),
                            "CV_scores_std": CV_scores.std(),
                            "test_accuracy": test_accuracy,
                        }, ignore_index=True)

                        classic_result_df.to_csv(result_dataset_folder + "classic_result_df.csv", index=True)

                    except:
                        traceback.print_exc()



            elif selection_algorithm_name in QUBO_algorithms_dict:

                # for alpha_input in [0.1, 0.3, 0.5, 0.7, 0.9, "balanced"]:
                for alpha_input in [0.5]:

                    for QUBO_solver_name, QUBO_solver in QUBO_solvers_dict.items():

                        if n_features>300 and QUBO_solver_name == "QPU":
                                continue

                        if alpha_input == "balanced":
                            alpha_heuristic = "balanced"
                            _, alpha_value = _get_BQM_alpha(selection_algorithm_instance._Q, alpha_input)
                        else:
                            alpha_heuristic = "hyperparameter"
                            alpha_value = alpha_input

                        classifier_folder = result_dataset_folder + selection_algorithm_name + "/" + QUBO_solver_name + "/"

                        if QUBO_solver_name == "QPU":
                            # QUBO_solver = LazyFixedEmbeddingComposite(QUBO_solver)
                            fixed_embedding, embedding_time = _get_fixed_embedding(QUBO_solver, n_features, classifier_folder)

                            sampler_hyperparams = {
                                "chain_strength": uniform_torque_compensation,
                                "num_reads": 100,
                                }

                            if not fixed_embedding:
                                print("Embedding failed, QUBO may be too big, n_features is {}.".format(n_features))
                                raise Exception

                            QUBO_solver = FixedEmbeddingComposite(QUBO_solver, fixed_embedding)

                        elif QUBO_solver_name == "QPUHybrid":
                            embedding_time = np.nan
                            sampler_hyperparams = {}
                        else:
                            embedding_time = np.nan
                            sampler_hyperparams = {"num_reads": 100}

                        # Check if it was already done
                        if QUBO_result_df[(QUBO_result_df["selection_algorithm_name"]==selection_algorithm_name) &
                                          (QUBO_result_df["target_feature_k"]==target_feature_k) &
                                          (QUBO_result_df["QUBO_solver"]==QUBO_solver_name) &
                                          (QUBO_result_df["alpha_value"].round(4) == round(alpha_value, 4))].empty:

                            print("Selection algorithm: {}, target number of features {}/{}, alpha: {}, QUBO solver: {}".format(selection_algorithm_name, target_feature_k, n_features, alpha_input, QUBO_solver_name))

                            try:
                                selected_features, sampleset_df, sampler_info, select_best_k_time = selection_algorithm_instance.select_best_k(target_feature_k, QUBO_solver, alpha_input, sampler_hyperparams = sampler_hyperparams)


                                X_train_selected_features = X_train[selected_features].copy()
                                X_test_selected_features = X_test[selected_features].copy()

                                CV_scores, test_accuracy, classifier_algorithm_fit_time, classifier_algorithm_CV_time = _evaluate_classifier(X_train_selected_features, Y_train, X_test_selected_features, Y_test, classifier_folder, target_feature_k)

                                QUBO_result_df = QUBO_result_df.append({
                                    "selection_algorithm_name": selection_algorithm_name,
                                    "selection_algorithm_fit_time": selection_algorithm_instance._fit_time,
                                    "selection_algorithm_select_best_k_time": select_best_k_time,
                                    "selected_features": selected_features,
                                    "target_feature_k": target_feature_k,
                                    "actual_feature_k": len(selected_features),
                                    "alpha_heuristic": alpha_heuristic,
                                    "alpha_value": alpha_value,
                                    "classifier_algorithm_name": "RandomForestClassifier",
                                    "classifier_algorithm_fit_time": classifier_algorithm_fit_time,
                                    "classifier_algorithm_CV_time": classifier_algorithm_CV_time,
                                    "CV_scores": CV_scores,
                                    "CV_scores_mean": CV_scores.mean(),
                                    "CV_scores_std": CV_scores.std(),
                                    "QUBO_solver": QUBO_solver_name,
                                    "embedding_time": embedding_time,
                                    "test_accuracy": test_accuracy,
                                    **sampler_info
                                }, ignore_index=True)

                                QUBO_result_df.to_csv(result_dataset_folder + "QUBO_result_df.csv", index=True)
                                sampleset_df.to_csv(classifier_folder + "/sampleset_df_k_{}_alpha_{}.csv".format(target_feature_k, alpha_input), index=True)

                            except:
                                traceback.print_exc()


    print("Dataset: {}, Complete!".format(dataset_name))






def run_dataset_experiment_parallel(dataset_name, result_root_folder, classic_algorithms_dict, QUBO_algorithms_dict, QUBO_solvers_dict):

    try:
        print("\n\nDataset name: {}".format(dataset_name))

        result_dataset_folder = os.path.join(result_root_folder, dataset_name + "/")

        # If directory does not exist, create
        if not os.path.exists(result_dataset_folder):
            os.makedirs(result_dataset_folder)

        run_dataset_experiment(dataset_name, result_dataset_folder, classic_algorithms_dict, QUBO_algorithms_dict, QUBO_solvers_dict)

        print_dataset_result_summary(result_dataset_folder)

    except Exception as e:
        print("On dataset {} Exception {}".format(dataset_name, str(e)))
        traceback.print_exc()


if __name__ == '__main__':


    DATASET_LIST = [
                    "waveform-5000",
                    "SPECTF",
                    "covertype",
                    "spambase",
                    "nomao",
                    "tecator",
                     "USPS",
                    "isolet",
                    "Bioresponse",
                    "SVHN_small",
                    "gisette",
                    ]


    # CLASSIC ALGORITHMS
    classic_algorithms_dict = {
        "ANOVA F Test": ANOVA_F_Test,
        "Chi2 Test": Chi2,
        "Linear Mutual Information MIToolbox": LinearMutualInformationMIT,
        "Linear Pearson Correlation": LinearPearsonCorrelation,
        "Linear SVC Boosting": LinearSVCBoosting,
        "Variance Threshold": VarianceThreshold,
    }


    # QUBO ALGORITHMS
    QUBO_algorithms_dict = {
        "QUBOCorrelation": QUBOCorrelation,
        "QUBOMutualInformation": QUBOMutualInformation,
        "QUBOSVCBoosting": QUBOSVCBoosting,
    }

    # Solvers
    QUBO_solvers_dict = {
        "SimulatedAnnealing": SimulatedAnnealingSampler(),
        "SteepestDescent": greedy.SteepestDescentSolver(),
        "TabuSampler": TabuSampler(),
    }

    result_root_folder = "results_classification/"



    run_dataset_experiment_parallel_partial = partial(run_dataset_experiment_parallel,
                                                      result_root_folder = result_root_folder,
                                                      classic_algorithms_dict = classic_algorithms_dict,
                                                      QUBO_algorithms_dict = QUBO_algorithms_dict,
                                                      QUBO_solvers_dict = QUBO_solvers_dict)



    pool = multiprocessing.Pool(processes=5, maxtasksperchild=1)
    resultList = pool.map(run_dataset_experiment_parallel_partial, DATASET_LIST, chunksize=1)

    pool.close()
    pool.join()


    sampler_QA = {"topology__type": "pegasus", "name__contains": "Advantage_system"}
    sampler_QA = DWaveSampler(client="qpu", solver=sampler_QA)
    sampler_QAHybrid = LeapHybridSampler()

    # Solvers
    QUBO_QPU_solvers_dict = {
        "QPU": sampler_QA,
        "QPUHybrid": sampler_QAHybrid,
    }

    run_dataset_experiment_parallel_partial = partial(run_dataset_experiment_parallel,
                                                      result_root_folder = result_root_folder,
                                                      classic_algorithms_dict = classic_algorithms_dict,
                                                      QUBO_algorithms_dict = QUBO_algorithms_dict,
                                                      QUBO_solvers_dict = QUBO_QPU_solvers_dict)

    for dataset_name in DATASET_LIST:
        try:
            run_dataset_experiment_parallel_partial(dataset_name)
        except Exception as e:
            print("On dataset {} Exception {}".format(dataset_name, str(e)))
            traceback.print_exc()


    print_global_result_summary(result_root_folder, DATASET_LIST)


