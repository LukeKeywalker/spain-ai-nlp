import csv
import pandas as pd


def load_dataset():
    column_names = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=column_names, header=None)[1:]
    return data


def sanitize_input(input):
    """
    TODO: share the same method between prepare-training-set.py and here
    :param input:
    :return:
    """
    if input.startswith('"') and input.endswith('"'):
        input = input[1:-1]

    return input \
        .replace('<br>', '') \
        .replace('</br>', '') \
        .replace('<br/>', '') \
        .replace('|', '') \
        .upper()


def save_dataframe_column(data, column_name, file_path, header=False):
    """
    Saves dataframe column to file
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
