import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import dataset_utils

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

def get_key_words(df):
    # do some cleaning of the input data
    data = df.copy()
    data['clean_name'] = data.name.apply(lambda x: dataset_utils.sanitize_input(x.lower()))
    data['clean_desc'] = data.description.apply(lambda x: dataset_utils.sanitize_input(x.lower()))
    corpus = list()
    corpus.append(data['clean_name'])
    corpus.append(data['clean_desc'])
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(corpus)
    #feature_names = tfidf.get_feature_names()
    return tfidf.get_feature_names()

def get_test_data():
    column_names = ['description']
    test_data = pd.read_csv('../data/test_descriptions.csv', names=column_names, header=None)[1:]
    return test_data

def get_corpus3(df):
    """
    Compute the text corpus from the dataframe after completing
    a full cleaning process, removing all except alphanumeric text
    """
    corpus = []
    for index, row in df.iterrows():
        corpus.append(clean_and_tokenize2(row['description']))
    #corpus = ' '.join(corpus)
    return corpus

def get_vectors_and_names():
    test_data = get_test_data()
    corpus = get_corpus3(test_data)
    #print(len(corpus))
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf_vectors = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names()
    return tfidf_vectors, feature_names

def get_top_words(tfidf_vectors, feature_names, position, n):
    '''
    It extracts the tifidf vector corresponding to the given position and returns the top n words
    '''
    vector_tfidfvectorizer=tfidf_vectors[position]
    df0 = pd.DataFrame(vector_tfidfvectorizer.T.todense(), index=feature_names,
                  columns=["tfidf"])
    df0 = df0.sort_values(by=["tfidf"],ascending=False)
    return df0[:n].index.tolist()

def get_top5_score(top_5, answer):
    '''
    Give the answer the score using one components:
    1. the matches with top_5 words
    Max. score = 1 (if there are 5 similar words)
    :param top_5: top 5 words for that answer
    :param answer: list of words of the answer
    '''
    answer = [item for sublist in answer for item in sublist]
    #print(type(answer))
    unique_words = set(answer)
    score = 0.0
    for word in unique_words:
        if word in top_5:
            score += 0.25
    #print("unique words {} and score {}".format(unique_words, score))
    return score

def get_tfidf_score(answer, top_5):
    '''
    This method will return a new column with the scores given for each
    answer in the list.
    :param answer: list of candidates
    :param top_5: top five words for that description in the test file
    '''
    # clean the answer the same way we did for the tfidf model
    #print(len(answer))
    #print("answer {}".format(answer))
    c_answer = clean_tokenize_answer(answer)

    # score only with similar words
    return get_top5_score(top_5, c_answer)

def get_global_score(top_5, answer):
    '''
    Give the answer the total score using two components:
    1. the matches with top_5 words
    2. the length/words_count score
    Max. score = 2
    :param top_5: top 5 words for that answer
    :param answer: list of words of the answer
    '''
    unique_words = set(answer)
    score = 0.0
    for word in unique_words:
        if word in top_5:
            score += 0.25
    score += length_wc_scoring(answer,0.3,0.7)
    return score

def new_sorted_list(row, tfidf_vectors, feature_names):
    scores = []
    candidates = row['name']
    row_index = row['row_num']
    row_list = candidates.split(",")
    # for each answer we have already a list of words
    # get the top 5 words for that answer
    top_5 = get_top_words(tfidf_vectors, feature_names, row_index, 5)
    #print("top_5 {}".format(top_5))
    for i in range(len(row_list)):
        r = row_list[i]

        # compute the score
        s = get_tfidf_score(r.strip(), top_5)
        #print("answer{} and score{}".format(r,s))
        scores.append(s)
    df = pd.DataFrame()
    df['answer'] = row_list
    df['score'] = scores
    #print(df)
    df = df.sort_values(by='score', ascending=False)
    #print(df)
    return ",".join(df.answer)

def get_sorted_submission(submission, to_csv):
    df = submission.copy()
    df['row_num'] = df.reset_index().index
    tfidf_vectors, feature_names = get_vectors_and_names()
    df['sorted'] = df.apply(lambda row: new_sorted_list(row, tfidf_vectors, feature_names), axis=1)
    df.head()
    df_top5 = df[['sorted']]
    df_top5.rename(columns={'sorted': 'name'}, inplace=True)
    # optional save the result in a csv file
    if to_csv:
        df_top5.to_csv('../submission/submission_sorted.csv', sep='\n', index=False)
    return df_top5