import gpt_2_simple as gpt2
import re

ANSWER_PATTERN = re.compile(r'<\|startofname\|>(.+?)<\|endofname\|>')
NUMBER_OF_ANSWERS_GENERATED = 100
NUMBER_OF_ANSWERS_SELECTED = 10


def generate_model_outputs(input):
    gpt_2_prompt = "<|startoftext|><|startofdesc|>{}<|endofdesc|><|startofname|>"
    description = gpt_2_prompt.format(input)
    return gpt2.generate(
        sess,
        temperature=1.0,
        length=40,
        nsamples=NUMBER_OF_ANSWERS_GENERATED,
        batch_size=10,
        prefix=description,
        run_name="spain-ai-nlp",
        return_as_list=True
    )


def extract_answer(output):
    matched_answers = re.findall(ANSWER_PATTERN, output)
    if len(matched_answers) > 0:
        return matched_answers[0].upper()
    else:
        return None


def answer_quality(answer, prompt):
    answer_word_list = answer.upper().split(" ")
    prompt_word_list = prompt.upper().split(" ")
    number_of_common_words = len(list(set(prompt_word_list) & set(answer_word_list)))

    return 1.0 - number_of_common_words / float(len(answer_word_list))


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(
    sess,
    checkpoint_dir='models',
    run_name='spain-ai-nlp'
)

while True:
    prompt = input("(>^_^)> ");
    outputs = generate_model_outputs(prompt)
    answers = sorted(
        list(
            dict.fromkeys(
                list(
                    filter(
                        lambda x: x is not None,
                        [extract_answer(output) for output in outputs if output is not None]
                    )
                )
            )),
        key=lambda x: answer_quality(x, prompt)
    )[:NUMBER_OF_ANSWERS_SELECTED]

    for (index, answer) in zip(range(1, len(answers) + 1), answers):
        print("{}. {}".format(index, answer))
