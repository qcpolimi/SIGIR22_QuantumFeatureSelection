import glob
import itertools
import os
import subprocess


def check_folder(path, folder):
    folder_path = f'{path}{folder}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def check_folders():
    letor_path = './results_ranking/'
    check_folder(letor_path, 'models')
    check_folder(letor_path, 'processed')


def check_dataset_folders(dataset_result_path):
    check_folder(dataset_result_path, 'time')
    check_folder(dataset_result_path, 'training_performance')
    check_folder(dataset_result_path, 'test_performance')


def run_letor_ranking(dataset_list, linear_fs_methods, qubo_fs_methods, qubo_solvers):
    check_folders()

    fs_methods = list(itertools.product(linear_fs_methods, [None]))
    fs_methods.extend(list(itertools.product(qubo_fs_methods, qubo_solvers)))

    for dataset in dataset_list:
        run_letor_ranking_dataset(dataset, fs_methods)


def run_letor_ranking_dataset(dataset, fs_methods, fold='Fold1', metric='NDCG@10', ranker='6'):
    print(f'Running the Ranking task for {dataset}...')

    dataset_data_path = f'./data/letor/{dataset}/'
    if dataset == 'OHSUMED':
        dataset_data_path += 'QueryLevelNorm/'
    dataset_result_path = f'./results_ranking/{dataset}/'
    check_dataset_folders(dataset_result_path)

    train_file = f'{dataset_data_path}{fold}/train.txt'
    val_file = f'{dataset_data_path}{fold}/vali.txt'
    test_file = f'{dataset_data_path}{fold}/test.txt'

    for fs_method, qubo_solver in fs_methods:
        print(f'[{fs_method}, {qubo_solver}]')

        base_fs_id = fs_method
        fs_path = f'{dataset_result_path}{fs_method}/'
        if qubo_solver is not None:
            base_fs_id += f' {qubo_solver}'
            fs_path += f'{qubo_solver}/'

        feature_paths = glob.glob(f'{fs_path}**/*.txt', recursive=True)
        for feature_path in feature_paths:
            feature_file = os.path.basename(feature_path)
            print(f'[{feature_file}]')

            fs_id = f'{base_fs_id}_{feature_file[:-4]}'
            out_id = f'{dataset}_{fold}_{fs_id}_{metric}'
            out_id = out_id.replace(' ', '-')

            model_file = f'./results_ranking/models/{out_id}.txt'
            train_perf_file = f'{dataset_result_path}training_performance/{out_id}.txt'
            time_file = f'{dataset_result_path}time/{out_id}.txt'

            print(f'[{out_id}] Fitting...')
            fit_args = ['bash', './Letor/fit_script.sh',
                        train_file, val_file, feature_path, metric, ranker, model_file, train_perf_file, time_file]
            subprocess.run(fit_args)

            print(f'[{out_id}] Testing...')
            test_performance_file = f'{dataset_result_path}test_performance/{out_id}'
            test_args = ['bash', './Letor/test_script.sh',
                         test_file, model_file, test_performance_file]
            subprocess.run(test_args)
