import argparse
import re
import os

import alive_progress as ap
import gpt_2_simple as gpt2
import pandas as pd

ANSWER_PATTERN = re.compile(r'<\|startofname\|>(.+?)<\|endofname\|>')
NUMBER_OF_ANSWERS_GENERATED = 10
NUMBER_OF_ANSWERS_SELECTED = 10
SUBMISSION_ROW = '{}\n'

# make TF less talkative
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_session = gpt2.start_tf_sess()


def generate_model_outputs(input):
    """
    Generates NUMBER_OF_ANSWERS_GENERATED answers
    using gpt-2 model loaded in TF Session from
    given input
    :param input: gpt-2 prompt (starting text)
    :return: list of model results
    """
    gpt_2_prompt = "<|startoftext|> <|startofdesc|> {} <|endofdesc|> <|startofname|>"
    description = gpt_2_prompt.format(sanitize_input(input))
    return gpt2.generate(
        tf_session,
        temperature=1.0,
        length=60,
        nsamples=NUMBER_OF_ANSWERS_GENERATED,
        batch_size=10,
        prefix=description,
        run_name="spain-ai-nlp",
        return_as_list=True,
        seed=666
    )


def sanitize_answer(item):
    """
    Cleans up answer given by a model.
    - removes commas (reserved for item name separation in submission file)
    - strips ending dots
    :param item:
    :return:
    """
    item = item             \
        .upper()            \
        .replace(',', '')   \
        .strip()

    if item.endswith('.'):
        return item[0:-1]
    else:
        return item


def sanitize_input(input):
    """
    TODO: share the same method between prepare-training-set.py and here
    :param input:
    :return:
    """
    if input.startswith('"') and input.endswith('"'):
        input = input[1:-1]

    input = input \
        .replace('<br>', '') \
        .replace('</br>', '') \
        .replace('<br/>', '') \
        .upper() \
        .replace('.', '') \
        .replace(',', '')

    return input


def extract_answer(model_output):
    """
    Extracts answer from model output
    using ANSWER_PATTERN regular expression
    :param model_output: single element of model generation result
    :return: extracted answer
    """
    matched_answers = re.findall(ANSWER_PATTERN, model_output)
    if len(matched_answers) > 0:
        return sanitize_answer(matched_answers[0])
    else:
        return None


def answer_quality(answer, prompt):
    """
    Calculates heuristics of answer quality in
    the context of prompt given to the model
    :param answer: answer extracted from model output
    :param prompt: prompt given to the model
    :return: float value between 0 and 1. 0 - answer considered nonsensical, 1 - answer considered exact
    """
    answer_word_list = answer.upper().split(" ")
    prompt_word_list = prompt.upper().split(" ")
    number_of_common_words = len(list(set(prompt_word_list) & set(answer_word_list)))

    return number_of_common_words / float(len(answer_word_list))


def without_duplicates(items):
    """
    Removes duplicates from given list
    :param items:
    :return: list of unique items
    """
    return list(dict.fromkeys(items))


def run_interactive():
    """
    Interactive loop for taking prompt from
    from the standard input and writing generated
    answers to standard output
    """
    while True:
        user_input = input("(>^_^)> ")
        outputs = generate_model_outputs(user_input)
        answers = select_answers(outputs, user_input)

        for (index, answer) in zip(range(1, len(answers) + 1), answers):
            print("{}. {}".format(index, answer))


def select_answers(outputs, prompt):
    """
    Returns top answers according to extract_answer heuristics
    :param outputs: output generated by the model
    :param prompt: prompt give to the model
    :return: list of top NUMBER_OF_ANSWERS SELECTED answers sorted according to extarct_answer quality heuristics
    """
    answers = sorted(
        without_duplicates(
            list(
                filter(
                    lambda x: x is not None,
                    [extract_answer(output) for output in outputs if output is not None]
                )
            )
        ),
        key=lambda x: answer_quality(x, prompt),
        reverse=True
    )[: NUMBER_OF_ANSWERS_SELECTED]
    return answers


def generate_item_name_candidates(description, step, on_finished):
    """
    Generates list of item name candidates sorted according to
    quality heuristics generated from given item description
    :param step: number of a step to be executed
    :param description: single description from test_descriptions.csv file
    :param tf_session: FT session with gpt-2 model loaded
    :param on_finished: callback on item generated
    :return:
    """
    reset_model(step)
    outputs = generate_model_outputs(description)
    answers = select_answers(outputs, description)
    on_finished()
    return ', '.join(answers)


def generate_submission():
    """
    Generates submission.csv file from test_descriptions.csv testing set
    """

    test_descriptions_file_path = 'data/test_descriptions.csv'
    submission_file_path = 'submission/submission.csv'

    test_descriptions = pd.read_csv(test_descriptions_file_path)

    descriptions = test_descriptions['description'].tolist()
    with ap.alive_bar(len(test_descriptions), bar='filling') as bar:
        names = [generate_item_name_candidates(description, index, bar)
                 for (index, description)
                 in zip(range(1, len(descriptions) + 1), descriptions)
                 ]

    with open(submission_file_path, 'w') as file:
        file.write(SUBMISSION_ROW.format('name'))
        for item_name_candidates in names:
            file.write(SUBMISSION_ROW.format(item_name_candidates))


def parse_arguments():
    global args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", required=False)
    return arg_parser.parse_args()


def load_model():
    gpt2.load_gpt2(
        tf_session,
        checkpoint_dir='models',
        run_name='spain-ai-nlp',
        multi_gpu=True
    )


def reset_model(step_count):
    global tf_session
    if step_count % 20 == 0:
        tf_session = gpt2.reset_session(sess=tf_session)
        load_model()


load_model()
args = parse_arguments()
if args.mode == 'interactive':
    run_interactive()
else:
    generate_submission()
