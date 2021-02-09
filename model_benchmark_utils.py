import pandas as pd
import numpy as np
from dataset_utils import load_test_names, load_test_descriptions


def discountedCumulativeGain(result, quality):
    acc = []
    for idx, val in enumerate([quality(x) for x in result]):
        ntor = 2 ** val - 1
        dtor = np.log2(idx + 2)
        score = ntor / dtor
        acc.append(score)
    return sum(acc)


def normalizedDiscountedCumulativeGain(result, quality):
    dcg = discountedCumulativeGain(result, quality)
    if dcg == 0:
        return 0
    idcg = discountedCumulativeGain(sorted(result, key=quality, reverse=True), quality)
    ndcg = dcg / idcg
    return ndcg


def create_test_set_dataframe(names, descriptions, answers):
    test_set_columns = ['name', 'description', 'answers']
    return pd.DataFrame(
        list(zip(names, descriptions, answers)),
        index=None, columns=test_set_columns
    )


def create_empty_test_set_dataframe():
    return create_test_set_dataframe([], [], [])


def load_answer_set_of_model(model, answer_set):
    answers_path = 'data/answers/model-{0}/{1}-answers.txt'
    return load_test_names(answers_path.format(model, answer_set))


def load_test_names_set(names_set):
    names_path = 'data/testsets/{0}-names.txt'
    return load_test_names(names_path.format(names_set))


def load_test_descriptions_set(descriptions_set):
    descriptions_path = 'data/testsets/{0}-descriptions.txt'
    return load_test_descriptions(descriptions_path.format(descriptions_set))


def load_scored_test_set(test_set, model):
    names = load_test_names_set(test_set)
    descriptions = load_test_descriptions_set(test_set)
    answers = load_answer_set_of_model(model, test_set)
    return names, descriptions, answers


def evaluate_test_set(test_set, sorting):
    (names, descriptions, answers) = test_set

    scored_test_set = create_test_set_dataframe(names, descriptions, answers)

    scored_test_set['score'] = scored_test_set[['name', 'description', 'answers']].apply(
        lambda x: discountedCumulativeGain(
            sorted(list(x['answers'].split(', ')), key=lambda name: sorting(name, x['description']), reverse=True),
            lambda y: 1 if y == x['name'].strip() else 0
        ),
        axis=1
    )

    test_set_score = scored_test_set['score'].sum() / float(len(scored_test_set))

    return scored_test_set, test_set_score


def evaluate_all_test_sets_of_model(model, sorting):
    test_set = create_empty_test_set_dataframe()
    for i in range(10):
        try:
            (scored_test_set, score) = evaluate_test_set(load_scored_test_set(i, model), sorting)
            test_set = test_set.append(scored_test_set)
        except:
            pass
    return test_set, test_set['score'].sum() / float(len(test_set))


def model_potential(model_answers):
    return len(model_answers[model_answers['score'] > 0]) / float(len(model_answers))
