import csv
import pandas as pd
import numpy as np


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


def discountedCumulativeGain(result, quality):
    acc = []
    for idx, val in enumerate([quality(x) for x in result]):
        ntor = 2**val - 1
        dtor = np.log2(idx + 2)
        score = ntor/dtor
        acc.append(score)
    return sum(acc)


def load_test_descriptions(test_descriptions_file_path):
    test_descriptions = pd.read_csv(test_descriptions_file_path, sep="~")
    return test_descriptions['description'].tolist()


def load_test_names(names_file_path):
    test_names = pd.read_csv(names_file_path, sep="~", dtype='string')
    return test_names['name'].tolist()


def normalizedDiscountedCumulativeGain(result, quality):
    dcg = discountedCumulativeGain(result, quality)
    if dcg == 0:
        return 0
    idcg = discountedCumulativeGain(sorted(result, key=quality, reverse=True), quality)
    ndcg = dcg / idcg
    return ndcg


