import pandas as pd
from nltk.tokenize import RegexpTokenizer

def get_nr_words(text):
    print(text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def length_wc_scoring(answer,l_max, w_max):
    """
    Gives a score (range [0,1]) to the answer
    considering two metrics:
    1. length of the answer: max score l_max
    2. number of words: max score w_max
    Taking the references observed in the train data
    :param answer: the answer to score
    :return: the score between 0 and 1
    """
    score = 0.0
    # length
    l = len(answer)
    # maximum score for length between 19 and 30 chars
    if l in range(19,31):
        score += l_max
    else: # inverse proportional to how far you are from the avg
        avg = 24
        score += l_max/abs(avg - l)
    # word count
    wc = get_nr_words(answer)
    # maximum score for 3, 4 or 5 words
    if wc in [3,4,5]:
        score += w_max
    else: # inverse proportional to how far you are from the avg
        avg = 4
        score += w_max/abs(avg - wc)
    return score