import os
import gpt_2_simple as gpt2

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, model_name=model_name, checkpoint_dir='checkpoint')


prompt = "<|startoftext|><|startofdesc|>{}<|endofdesc|><|startofname|>"
while True:
    description = prompt.format(input(":> "))
    gpt2.generate(
        sess,
        length=40,
        nsamples=10,
        batch_size=10,
        prefix=description,
        run_name="spain-ai-nlp"
    )
