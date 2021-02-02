import gpt_2_simple as gpt2


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(
    sess,
    checkpoint_dir='models',
    run_name='spain-ai-nlp'
)


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
