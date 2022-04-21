import pandas as pd

from utils.DataIO import DataIO


class LetorLoader:
    DATASET_LIST = ['2003_hp_dataset', '2003_np_dataset', '2003_td_dataset', '2004_hp_dataset', '2004_np_dataset',
                    '2004_td_dataset', 'OHSUMED', 'MQ2007', 'MQ2008']
    FOLD_LIST = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']

    def __init__(self, folder_path, dataset_name, fold='Fold1'):
        assert dataset_name in self.DATASET_LIST, \
            f'Please provide a valid dataset name from the following list:\n{self.DATASET_LIST}.'
        assert fold in self.FOLD_LIST, f'Please provide a valid fold from the following list:\n{self.FOLD_LIST}.'

        dataIO = DataIO(folder_path + 'data/')

        try:
            data_split = dataIO.load_data(fold)

            for attrib_name, attrib_object in data_split.items():
                self.__setattr__(attrib_name, attrib_object)

            print(f'Loaded Letor {dataset_name} from experiment files.')

        except FileNotFoundError:
            print(f'Loading Letor {dataset_name} from original files...')

            if dataset_name in ['2003_hp_dataset', '2003_np_dataset', '2003_td_dataset', '2004_hp_dataset',
                                '2004_np_dataset', '2004_td_dataset']:
                self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = load_letor3_gov(
                    dataset_name, fold=fold)
            elif dataset_name == 'OHSUMED':
                self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = load_letor3_ohsumed(fold)
            elif dataset_name == 'MQ2007':
                self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = load_letor4_mq2007(fold)
            elif dataset_name == 'MQ2008':
                self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = load_letor4_mq2008(fold)

            data_split = {
                'X_train': self.X_train,
                'X_val': self.X_val,
                'X_test': self.X_test,
                'Y_train': self.Y_train,
                'Y_val': self.Y_val,
                'Y_test': self.Y_test,
            }

            dataIO.save_data(fold, data_split)

            print(f'Saved Letor {dataset_name} in experiment files.')


def load_letor(letor):
    classes = letor.pop(letor.columns[0]).to_numpy(dtype=int)
    letor = letor.drop(letor.columns[0], axis=1)

    letor.columns = letor.columns - 1

    for col in range(1, len(letor.columns) + 1):
        letor.loc[:, col] = letor.loc[:, col].apply(lambda x: float(str(x).split(':')[1]))

    return letor, classes


def load_letor3_split(dataset_path, split='train.txt'):
    if split[-4:] != '.txt':
        split += '.txt'

    dataset_path += split
    letor = pd.read_csv(dataset_path, sep=" ", header=None)
    letor = letor.drop(letor.columns[-3:], axis=1)

    return load_letor(letor)


def load_letor3(dataset_path):
    X_train, Y_train = load_letor3_split(dataset_path, split='train.txt')
    X_val, Y_val = load_letor3_split(dataset_path, split='vali.txt')
    X_test, Y_test = load_letor3_split(dataset_path, split='test.txt')
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_letor4_split(dataset_path, split='train.txt'):
    if split[-4:] != '.txt':
        split += '.txt'

    dataset_path += split
    letor = pd.read_csv(dataset_path, sep=" ", header=None)
    letor = letor.drop(letor.columns[-9:], axis=1)

    return load_letor(letor)


def load_letor4(dataset_path):
    X_train, Y_train = load_letor4_split(dataset_path, split='train.txt')
    X_val, Y_val = load_letor4_split(dataset_path, split='vali.txt')
    X_test, Y_test = load_letor4_split(dataset_path, split='test.txt')
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_letor3_gov(version, fold='Fold1'):
    return load_letor3(f'./data/letor/Gov/QueryLevelNorm/{version}/{fold}/')


def load_letor3_ohsumed(fold='Fold1'):
    return load_letor3(f'./data/letor/OHSUMED/QueryLevelNorm/{fold}/')


def load_letor4_mq2007(fold='Fold1'):
    return load_letor4(f'./data/letor/MQ2007/{fold}/')


def load_letor4_mq2008(fold='Fold1'):
    return load_letor4(f'./data/letor/MQ2008/{fold}/')
