import glob
import os

import pandas as pd
import tqdm


def dataset_result_dataframe(dataset, result_root_folder):
    timefiles = glob.glob(f"{result_root_folder}/{dataset}/time/**/*.txt", recursive=True)
    trainginfiles = glob.glob(f"{result_root_folder}/{dataset}/training_performance/**/*.txt", recursive=True)
    testfiles = glob.glob(f"{result_root_folder}/{dataset}/test_performance/**/*.txt", recursive=True)

    times_df = []
    for f in timefiles:
        line = os.path.splitext(os.path.basename(f))[0].split("_")
        tdf = pd.read_csv(f)
        times_df.append(
            [line[k] for k in [0, 1, 2, 6, -1]] + [float(tdf.iloc[0]['end_time']) - float(tdf.iloc[0]['start_time'])])

    times_df = pd.DataFrame(times_df, columns=['dataset', 'fold', 'featSelector', 'features', 'measure', 'time'])

    trn_df = []
    for f in trainginfiles:
        line = os.path.splitext(os.path.basename(f))[0].split("_")
        with open(f, "r") as F:
            lf = F.readlines()
            val_score = lf[-4].strip().split(": ")[-1]
            train_score = lf[-5].strip().split(": ")[-1]
            trn_df.append([line[k] for k in [0, 1, 2, 6, -1]] + [float(train_score), float(val_score)])

    trn_df = pd.DataFrame(trn_df, columns=['dataset', 'fold', 'featSelector', 'features', 'measure', 'train_score',
                                           'val_score'])

    test_df = []
    for f in tqdm.tqdm(testfiles):
        line = os.path.splitext(os.path.basename(f))[0].split("_")
        with open(f, "r") as F:
            lf = F.readlines()
            test_score = lf[-1].strip().split(": ")[-1]
            test_df.append([line[k] for k in [0, 1, 2, 6, -2, -1]] + [float(test_score)])

    test_df = pd.DataFrame(test_df,
                           columns=['dataset', 'fold', 'featSelector', 'features', 'measure', 'perf', 'test_score'])
    test_df = test_df.pivot(values='test_score', index=['dataset', 'fold', 'featSelector', 'features', 'measure'],
                            columns='perf').reset_index()

    df = trn_df.merge(times_df, how='inner', on=['dataset', 'fold', 'featSelector', 'features', 'measure'])
    df = df.merge(test_df, how='inner', on=['dataset', 'fold', 'featSelector', 'features', 'measure'])

    df.to_csv(f"{result_root_folder}/processed/{dataset}_eval.csv", index=False)


def letor_result_dataframe(dataset_list, result_root_folder):
    for dataset in dataset_list:
        dataset_result_dataframe(dataset, result_root_folder)
