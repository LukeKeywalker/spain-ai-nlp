import os

import gpt_2_simple as gpt2

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              'data/trainset.txt',
              model_name=model_name,
              # restore_from='fresh',
              run_name='spain-ai-nlp',
              save_every=100,
              steps=9000)  # steps is max number of training steps
