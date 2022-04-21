#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2021

@author: Maurizio Ferrari Dacrema
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


def print_dataset_result_summary(result_dataset_folder):

    all_features_result_df = pd.read_csv(result_dataset_folder + "/all_features_result_df.csv", index_col=0)
    all_features_result_df = all_features_result_df.set_index("selection_algorithm_name")

    classic_result_df = pd.read_csv(result_dataset_folder + "/classic_result_df.csv", index_col=0)
    best_result_index = classic_result_df.groupby(['selection_algorithm_name'], sort=False)['CV_scores_mean'].idxmax()
    best_classic_result_df = classic_result_df.loc[best_result_index]
    best_classic_result_df = best_classic_result_df.set_index("selection_algorithm_name")

    QUBO_result_df = pd.read_csv(result_dataset_folder + "/QUBO_result_df.csv", index_col=0)
    QUBO_result_df = QUBO_result_df[QUBO_result_df["alpha_value"] == 0.5]
    best_result_index = QUBO_result_df.groupby(['selection_algorithm_name', "QUBO_solver"], sort=False)['CV_scores_mean'].idxmax()
    best_QUBO_result_df = QUBO_result_df.loc[best_result_index]
    best_QUBO_result_df = best_QUBO_result_df.set_index(['selection_algorithm_name', "QUBO_solver"])


    for col_label in ["classifier_algorithm_CV_time", "selection_algorithm_select_best_k_time"]:
        best_classic_result_df[col_label] = classic_result_df.groupby(['selection_algorithm_name'], sort=False)[col_label].sum()
        best_QUBO_result_df[col_label] = QUBO_result_df.groupby(['selection_algorithm_name', "QUBO_solver"], sort=False)[col_label].sum()

    best_QUBO_result_df = best_QUBO_result_df.reset_index()
    best_classic_result_df = best_classic_result_df.reset_index()

    result_summary = pd.concat([best_classic_result_df, best_QUBO_result_df, all_features_result_df], ignore_index=True)
    result_summary.to_csv(result_dataset_folder + "result_dataset_summary.csv", index=True)

    result_all = pd.concat([classic_result_df, QUBO_result_df, all_features_result_df], ignore_index=True)
    result_all.to_csv(result_dataset_folder + "result_dataset_all.csv", index=True)

    from matplotlib.pyplot import cm

    # Turn interactive plotting off
    plt.ioff()

    # Ensure it works even on SSH
    plt.switch_backend('agg')

    plt.xlabel('N Selected Features')
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per number of selected feature and algorithm")


    # marker_iterator = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))

    # color = itertools.cycle(('o', 'v', '^', '<', '>', 's', '8', 'p'))
    n_lines = len(classic_result_df['selection_algorithm_name'].unique()) + len(QUBO_result_df['selection_algorithm_name'].unique())*len(QUBO_result_df["QUBO_solver"].unique())
    color_iterator = iter(cm.tab20(np.linspace(0, 1, n_lines)))

    classic_result_df = classic_result_df.sort_values(by='actual_feature_k', ascending=True)
    QUBO_result_df = QUBO_result_df.sort_values(by='actual_feature_k', ascending=True)

    for selection_algorithm in classic_result_df['selection_algorithm_name'].unique():
        algorithm_results = classic_result_df[classic_result_df['selection_algorithm_name'] == selection_algorithm]
        plt.plot(algorithm_results["actual_feature_k"], algorithm_results["test_accuracy"], linewidth=2, label=selection_algorithm, color = next(color_iterator))


    for selection_algorithm in QUBO_result_df['selection_algorithm_name'].unique():
        color = next(color_iterator)
        marker_iterator = itertools.cycle(['o', 'v', '^', '<', '>', 's', '8', 'p'][:len(QUBO_result_df["QUBO_solver"].unique())])
        for QUBO_solver in QUBO_result_df["QUBO_solver"].unique():
            algorithm_results = QUBO_result_df[(QUBO_result_df['selection_algorithm_name'] == selection_algorithm) &
                                               (QUBO_result_df['QUBO_solver'] == QUBO_solver)]

            if "alpha_value" in algorithm_results.columns:
                algorithm_results = algorithm_results[algorithm_results["alpha_value"] == 0.5]

            plt.plot(algorithm_results["actual_feature_k"], algorithm_results["test_accuracy"],
                     linewidth=2, marker=next(marker_iterator), markersize = 6, color = color,
                     label=selection_algorithm + " " + QUBO_solver)

    plt.legend(bbox_to_anchor=(1.0, 1.0))

    plt.savefig(result_dataset_folder + "plot_test_accuracy.png", dpi = 600, bbox_inches='tight')



def print_global_result_summary(result_root_folder, dataset_list):

    global_result_df = None

    for dataset_name in dataset_list:

        result_dataset_folder = result_root_folder + dataset_name + "/"

        print_dataset_result_summary(result_dataset_folder)

        result_all_dataset = pd.read_csv(result_dataset_folder + "/result_dataset_summary.csv", index_col=0)
        result_all_dataset["dataset"] = dataset_name
        result_all_dataset.drop(columns = ['num_steps'], inplace=True, errors = "ignore")

        if global_result_df is None:
            global_result_df = result_all_dataset
        else:
            global_result_df = pd.concat([global_result_df, result_all_dataset], ignore_index=True)

    global_result_df.to_csv(result_root_folder + "result_global_summary.csv", index=True)

    global_result_table = global_result_df[['dataset','selection_algorithm_name',
                                            'actual_feature_k', 'QUBO_solver', 'test_accuracy',
                                            "alpha_value",
                                            "alpha_heuristic",
                                            'selection_algorithm_fit_time',
                                            'selection_algorithm_select_best_k_time',
                                            "target_feature_k", 'classifier_algorithm_fit_time',
                                            'CV_scores_mean', 'CV_scores_std']].copy()

    # global_result_table = global_result_df[['dataset','selection_algorithm_name',
    #                                         'actual_feature_k', 'QUBO_solver', 'test_accuracy',
    #                                         "alpha_value",
    #                                         "alpha_heuristic",
    #                                         "target_feature_k",
    #                                         'CV_scores_mean',
    #                                         'CV_scores_std']]

    global_result_table.to_csv(result_root_folder + "result_global_table.csv", index=True)
    global_result_table.fillna({'selection_algorithm_name': "All Features"}, inplace = True)

    ##############################################################################################################
    #################
    #################           Table with results for all QUBO solvers
    #################

    for QUBO_selection_algorithm_name in global_result_table[['QUBO_solver', 'selection_algorithm_name']].dropna()['selection_algorithm_name'].unique():

        selection_algorithm_result = global_result_table[global_result_table['selection_algorithm_name'] == QUBO_selection_algorithm_name]

        latex_table = pd.pivot_table(selection_algorithm_result, values=['actual_feature_k', "test_accuracy"], index=['dataset'], columns=['QUBO_solver'], aggfunc=np.max, fill_value=-1)
        latex_table = latex_table.swaplevel(0, 1, 1).sort_index(1)

        latex_table["All Features", "actual_feature_k"] = 0
        latex_table["All Features", "test_accuracy"] = 0

        for dataset in latex_table.index:
            all_f = global_result_table[(global_result_table["dataset"] == dataset) & (global_result_table["selection_algorithm_name"] == "All Features")].squeeze()
            latex_table.loc[dataset, ("All Features", "actual_feature_k")] = all_f["actual_feature_k"]
            latex_table.loc[dataset, ("All Features", "test_accuracy")] = all_f["test_accuracy"]

        latex_table.iloc[:, latex_table.columns.get_level_values(1)=='actual_feature_k'] = latex_table.iloc[:, latex_table.columns.get_level_values(1)=='actual_feature_k'].astype(int)
        latex_table.replace(-1, '', inplace = True)
        latex_table.rename(columns={"actual_feature_k":"N", "test_accuracy": "Accuracy"}, inplace=True)
        latex_table.rename_axis([None], inplace=True, axis=0)
        latex_table.rename_axis([None, None], inplace=True, axis=1)

        latex_table.sort_values(by=("All Features", "N"), inplace=True)
        # latex_table.sort_index(axis=1, inplace=True)
        columns = list(latex_table.columns.get_level_values(0).unique())
        columns.sort()
        columns.remove("All Features")
        latex_table = latex_table[["All Features", *columns]]

        latex_table.to_latex(result_root_folder + QUBO_selection_algorithm_name + ".txt",
            index = True,
            escape = True, #do not automatically escape special characters
            multicolumn = True,
            multicolumn_format = "c",
            column_format = "l|" + "rc|"*len(latex_table.columns.get_level_values(1)),
            float_format = "{:.4f}".format,
            )




    ##############################################################################################################
    #################
    #################           Linear and quadratic solvers
    #################

    latex_table = global_result_table[(global_result_table["QUBO_solver"] == "TabuSampler") | global_result_table["QUBO_solver"].isna()]

    latex_table = pd.pivot_table(latex_table, values=['actual_feature_k', "test_accuracy"], index=['dataset'],
                                 columns=['selection_algorithm_name'], aggfunc=np.max, fill_value=-1)
    latex_table = latex_table.swaplevel(0, 1, 1).sort_index(1)

    latex_table.iloc[:, latex_table.columns.get_level_values(1)=='actual_feature_k'] = latex_table.iloc[:, latex_table.columns.get_level_values(1)=='actual_feature_k'].astype(int)
    latex_table.replace(-1, '', inplace = True)
    latex_table.rename(columns={"actual_feature_k":"N", "test_accuracy": "Accuracy"}, inplace=True)
    latex_table.rename_axis([None], inplace=True, axis=0)
    latex_table.rename_axis([None, None], inplace=True, axis=1)

    latex_table.sort_values(by=("All Features", "N"), inplace=True)
    # latex_table.sort_index(axis=1, inplace=True)

    columns = list(latex_table.columns.get_level_values(0).unique())
    columns.sort()
    columns.remove("All Features")
    latex_table = latex_table[["All Features", *columns]]

    latex_table.rename(columns={"Linear Mutual Information": "Linear MI",
                                "Linear Pearson Correlation": "Linear Pearson"}, inplace=True)

    latex_table.to_latex(result_root_folder + "linear_and_quadratic_models.txt",
        index = True,
        escape = True, #do not automatically escape special characters
        multicolumn = True,
        multicolumn_format = "c",
        column_format = "l|" + "rc|"*len(latex_table.columns.get_level_values(1)),
        float_format = "{:.4f}".format,
        )


def letor_result_dataset_summary(result_dataset_folder):
    all_features_result_df = pd.read_csv(result_dataset_folder + "/all_features_result_df.csv", index_col=0)
    all_features_result_df = all_features_result_df.set_index("selection_algorithm_name")

    classic_result_df = pd.read_csv(result_dataset_folder + "/classic_result_df.csv", index_col=0)
    classic_result_df = classic_result_df.set_index("selection_algorithm_name")

    QUBO_result_df = pd.read_csv(result_dataset_folder + "/QUBO_result_df.csv", index_col=0)
    QUBO_result_df = QUBO_result_df.set_index(['selection_algorithm_name', "QUBO_solver"])

    for col_label in ["classifier_algorithm_CV_time", "selection_algorithm_select_best_k_time"]:
        classic_result_df[col_label] = classic_result_df.groupby(['selection_algorithm_name'], sort=False)[
            col_label].sum()
        QUBO_result_df[col_label] = \
            QUBO_result_df.groupby(['selection_algorithm_name', "QUBO_solver"], sort=False)[col_label].sum()

    classic_result_df = classic_result_df.reset_index()
    QUBO_result_df = QUBO_result_df.reset_index()

    result_summary = pd.concat([classic_result_df, QUBO_result_df, all_features_result_df], ignore_index=True)
    result_summary.to_csv(result_dataset_folder + "result_dataset_summary.csv", index=True)


def letor_global_result_summary(result_root_folder, dataset_list):
    global_result_df = None
    for dataset_name in dataset_list:
        result_dataset_folder = result_root_folder + dataset_name + "/"

        letor_result_dataset_summary(result_dataset_folder)

        result_all_dataset = pd.read_csv(result_dataset_folder + "/result_dataset_summary.csv", index_col=0)
        result_all_dataset["dataset"] = dataset_name
        result_all_dataset.drop(columns=['num_steps'], inplace=True, errors="ignore")

        if global_result_df is None:
            global_result_df = result_all_dataset
        else:
            global_result_df = pd.concat([global_result_df, result_all_dataset], ignore_index=True)

    global_result_df.to_csv(result_root_folder + "result_global_summary.csv", index=True)

    global_result_table = global_result_df[['dataset', 'selection_algorithm_name',
                                            'actual_feature_k', 'QUBO_solver', 'test_accuracy',
                                            "alpha_value",
                                            "alpha_heuristic",
                                            'selection_algorithm_fit_time',
                                            'selection_algorithm_select_best_k_time',
                                            "target_feature_k", 'classifier_algorithm_fit_time',
                                            'CV_scores_mean', 'CV_scores_std']].copy()

    global_result_table.to_csv(result_root_folder + "result_global_table.csv", index=True)


















