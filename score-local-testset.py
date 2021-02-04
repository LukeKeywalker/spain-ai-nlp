import argparse
import utils
import pandas as pd

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("benchmark_set", default=0)
arg_parser.add_argument("--model", default=2)

args = arg_parser.parse_args()

descriptions_path = 'data/testsets/{0}-descriptions.txt'
names_path = 'data/testsets/{0}-names.txt'
answers_path = 'data/answers/model-{0}/{1}-answers.txt'

descriptions = utils.load_test_descriptions(descriptions_path.format(args.benchmark_set))
names = utils.load_test_names(names_path.format(args.benchmark_set))
answers = utils.load_test_names(answers_path.format(args.model, args.benchmark_set))

scored_test_set = pd.DataFrame(
    list(zip(names, descriptions, answers)),
    index=None, columns=['name', 'description', 'answers']
)

scored_test_set['score'] = scored_test_set[['name', 'answers']].apply(
    lambda x: utils.discountedCumulativeGain(
        x['answers'].split(', '),
        lambda y: 1 if y == x['name'].strip() else 0
    ),
    axis=1
)

pd.set_option('display.max_rows', 100)
print(scored_test_set[['name', 'description', 'answers', 'score']].sort_values(by='score', ascending=False))
print('Total score: {}%'.format(
    scored_test_set['score'].sum() / float(len(scored_test_set))
))

