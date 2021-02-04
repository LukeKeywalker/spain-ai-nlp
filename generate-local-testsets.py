import pandas as pd
import csv

CURRENT_RANDOM_STATE = 0
NUMBER_OF_RANDOM_SAMPLES = 100


def load_dataset():
    colnames = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=colnames, header=None)[1:]
    return data


def select_random_samples(sample_count, data):
    """
    Selects sample_count number of random samples from a dataframe
    :param sample_count: number of samples selected
    :param data: dataframe
    :return: dataframe with random samples
    """
    global CURRENT_RANDOM_STATE
    CURRENT_RANDOM_STATE = CURRENT_RANDOM_STATE + 1
    return data.sample(n=sample_count, random_state=CURRENT_RANDOM_STATE)


def save_dataframe_column(data, column_name, file_path, header=False):
    """
    Saves dataframe column to fil
    :param file_path: path of the file to save
    :param data: dataframe
    :param column_name: name of the column to save
    :param header: include header flag
    :return:
    """
    data[column_name].to_csv(
        file_path,
        sep="@",
        doublequote=False,
        quoting=csv.QUOTE_NONE,
        header=header,
        index=False,
        quotechar="",
        escapechar="\\"
    )


dataset = load_dataset()
for i in range(100):
    random_samples = select_random_samples(NUMBER_OF_RANDOM_SAMPLES, dataset)
    save_dataframe_column(
        data=random_samples,
        column_name='name',
        file_path='data/local-testsets/{}-names.txt'.format(i),
        header=True)
    save_dataframe_column(
        data=random_samples,
        column_name='description',
        file_path='data/local-testsets/{}-descriptions.txt'.format(i),
        header=True
    )
