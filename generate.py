import argparse
import re
from dataset_utils import *

import alive_progress as ap
import gpt_2_simple as gpt2

ANSWER_PATTERN = re.compile(r'<\|startofname\|>(.+?)<\|endofname\|>')
NUMBER_OF_ANSWERS_GENERATED = 64
NUMBER_OF_ANSWERS_SELECTED = 64
SUBMISSION_ROW = '{}\n'
THREAD_COUNT = 8

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # make TF less talkative


def select_cuda_device(device_index=0):
    """

    :param device_index:
    :return:
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)


def generate_model_outputs(model, input):
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
        batch_size=32,
        prefix=description,
        run_name="model-{}".format(model),
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
    item = item \
        .upper() \
        .replace(',', '') \
        .strip()

    if item.endswith('.'):
        return item[0:-1]
    else:
        return item


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
    return 1.0


def without_duplicates(items):
    """
    Removes duplicates from given list
    :param items:
    :return: list of unique items
    """
    return list(dict.fromkeys(items))


def run_interactive(model):
    """
    Interactive loop for taking prompt from
    from the standard input and writing generated
    answers to standard output
    """
    while True:
        user_input = input("(>^_^)> ")
        outputs = generate_model_outputs(model, user_input)
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


def generate_item_name_candidates(model, description, step, on_finished, checkpoint_directory='checkpoint'):
    """
    Generates list of item name candidates sorted according to
    quality heuristics generated from given item description
    :param model: model version suffix
    :param step: number of a step to be executed
    :param description: single description from test_descriptions.csv file
    :param on_finished: callback on item generated
    :param checkpoint_directory: checkpoint directory to load model from
    :return:
    """
    reset_model(model, step, checkpoint_directory)
    outputs = generate_model_outputs(model, description)
    answers = select_answers(outputs, description)
    on_finished()
    return ', '.join(answers)


def generate_answers_file(model, test_descriptions_file_path, answer_file_path, checkpoint_directory='checkpoint'):
    """
    Writes names generated by model to a file
    """
    descriptions = load_test_descriptions(test_descriptions_file_path)
    with ap.alive_bar(len(descriptions), bar='filling') as bar:
        names = [generate_item_name_candidates(model, description, index, bar, checkpoint_directory)
                 for (index, description)
                 in enumerate(descriptions)]

    with open(answer_file_path, 'w') as file:
        file.write(SUBMISSION_ROW.format('name'))
        for item_name_candidates in names:
            file.write(SUBMISSION_ROW.format(item_name_candidates))


def parse_arguments():
    global args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", required=False)
    arg_parser.add_argument("--benchmark", required=False)
    arg_parser.add_argument("--gpu", required=False, default=0)
    arg_parser.add_argument("--model", required=False, default=2)
    return arg_parser.parse_args()


def load_model(model, checkpoint_directory='checkpoint'):
    global tf_session
    gpt2.load_gpt2(
        tf_session,
        checkpoint_dir=checkpoint_directory,
        run_name='model-{}'.format(model),
        multi_gpu=False
    )


def reset_model(model, step_count, checkpoint_directory='checkpoint'):
    global tf_session
    if step_count % 10 == 0:
        tf_session = gpt2.reset_session(sess=tf_session)
        load_model(model, checkpoint_directory)


def parse_benchmark_set(benchmark_items):
    return benchmark_items.split(',')


def generate_benchmark_answers(model, benchmark_set):
    """
    Generates item names for descriptions located in data/local-testset/
    and compares them with their respective names
    """
    try_create_dir('data/answers/')
    try_create_dir('data/answers/model-{}'.format(model))
    for benchmark in benchmark_set:
        test_descriptions_file_path = 'data/testsets/{}-descriptions.txt'.format(benchmark)

        answer_file_path = 'data/answers/model-{}/{}-answers.txt'.format(model, benchmark)
        generate_answers_file(model, test_descriptions_file_path, answer_file_path)


def init_model_session(model, checkpoint_directory='checkpoint'):
    global tf_session
    tf_session = gpt2.start_tf_sess()
    load_model(model, checkpoint_directory)


if __name__ == '__main__':
    args = parse_arguments()
    select_cuda_device(args.gpu)
    init_model_session(args.model)
    if args.mode == 'interactive':
        run_interactive(args.model)
    elif args.benchmark is not None:
        benchmark_set = parse_benchmark_set(args.benchmark)
        generate_benchmark_answers(args.model, benchmark_set)
    else:
        generate_answers_file(args.model, 'data/test_descriptions.csv', 'submission/submission.csv')
