import csv
import os

import pandas as pd


def load_dataset():
    """
    Loads unprocessed challenge training set
    :return: dataframe with name and description
    """
    column_names = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=column_names, header=None)[1:]
    return data


def sanitize_input(input):
    """
    Removes break line html task and | symbol from input string
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


def load_test_descriptions(test_descriptions_file_path):
    """
    :param test_descriptions_file_path: path to the single column CSV file containing item descriptions
    :return: list of item descriptions
    """
    test_descriptions = pd.read_csv(test_descriptions_file_path, sep="~")
    return test_descriptions['description'].tolist()


def load_test_names(names_file_path):
    """
    :param names_file_path: path to the single column CSV containing item names
    :return:
    """
    test_names = pd.read_csv(names_file_path, sep="~", dtype='string')
    return test_names['name'].tolist()


def try_create_dir(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError as e:
        pass
