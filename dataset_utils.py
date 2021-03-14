import csv
import os
import re
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer

def load_dataset():
    """
    Loads unprocessed challenge training set
    :return: dataframe with name and description
    """
    column_names = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=column_names, header=None)[1:]
    return data

def clean_and_tokenize(input):
    """
    Cleans the input and returns a list of tokens (words).
    Step needed to perform a tfidf analysis
    """
    if input.startswith('"') and input.endswith('"'):
        input = input[1:-1]

    input = input.replace('<br>', '') \
        .replace('</br>', '') \
        .replace('<br/>', '') \
        .replace('|', '') \
        .upper()
    words = re.sub(r"[^A-Za-z0-9\-]", " ", input).lower()
    return words


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

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def clean_tokenize_answer(answer):
    '''
    Method to remove non alphanumeric characters, to transform to lowercase
    and to stem the words. It returns the list of words per answer.
    '''
    list_answers = answer.split(',')
    clean_answers = []
    for a in list_answers:
        ca = clean_and_tokenize(a)
        ca = tokenize(ca)
        clean_answers.append(ca)
    return clean_answers

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
