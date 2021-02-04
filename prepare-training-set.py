import pandas as pd
import csv
import utils


def load_dataset():
    colnames = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=colnames, header=None)[1:]
    return data


def process_dataset(data):
    sample = '<|startoftext|> <|startofdesc|> {0} <|endofdesc|> <|startofname|> {1} <|endofname|> <|endoftext|>'
    data['processed'] = data.apply(
        lambda row: sample.format(utils.sanitize_input(row['description']), row['name']),
        axis=1
    )
    return data


dataset = process_dataset(load_dataset())
utils.save_dataframe_column(
    data=dataset,
    column_name='processed',
    file_path='data/trainset.txt'
)
