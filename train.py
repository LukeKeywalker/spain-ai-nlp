import argparse
import os

import gpt_2_simple as gpt2


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", required=False, default=2)
    return arg_parser.parse_args()


model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

args = parse_arguments()
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              model_name=model_name,
              dataset='data/trainset.txt',
              run_name='model-{}'.format(args.model),
              learning_rate=0.0002,
              save_every=1000,
              multi_gpu=True,
              sample_every=1000,
              batch_size=10,
              accumulate_gradients=False,
              use_memory_saving_gradients=True,
              steps=10000)
