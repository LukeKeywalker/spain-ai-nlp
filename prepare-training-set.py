import pandas as pd
import csv


def load_dataset():
    colnames = ['name', 'description']
    data = pd.read_csv('data/train.csv', names=colnames, header=None)[1:]
    return data


def sanitize(sample):
    if sample.startswith('"') and sample.endswith('"'):
        sample = sample[1:-1]

    return sample \
        .replace('<br>', '') \
        .replace('</br>', '') \
        .replace('<br/>', '') \
        .replace('|', '') \
        .upper()


def process_dataset(data):
    sample = '<|startoftext|> <|startofdesc|> {0} <|endofdesc|> <|startofname|> {1} <|endofname|> <|endoftext|>'
    data['processed'] = data.apply(
        lambda row: sample.format(sanitize(row['description']), row['name']),
        axis=1
    )
    return data


dataset = process_dataset(load_dataset())
dataset['processed'].to_csv(
    'data/trainset.txt',
    sep="@",
    doublequote=False,
    quoting=csv.QUOTE_NONE,
    header=False,
    index=False,
    quotechar="",
    escapechar="\\"
)
