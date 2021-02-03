import os

import gpt_2_simple as gpt2

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              dataset='data/trainset.txt',
              run_name='model-2',
              # restore_from='fresh',
              learning_rate=0.0002,
              save_every=1000,
              multi_gpu=True,
              batch_size=2,
              accumulate_gradients=False,
              use_memory_saving_gradients=True,
              steps=10000)
