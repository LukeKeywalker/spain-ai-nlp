# spain-ai-nlp
Solution of Spain AI NLP Challenge 2020

## Requirements:

* Python 3 installed
* `pip3 install gpt-2-simple`
* `pip3 install tensorflow==1.15` or ` pip3 install tensorflow-gpu==1.15` for CUDA enabled GPUs
* `pip3 install pandas`
* `pip3 install numpy`
* `pip3 install nltk`
* `pip3 install seaborn`
* `pip3 install alive-progress`

## Installing finetuned GPT-2 model
* [Get finetuned models here](https://drive.google.com/drive/folders/1AYZdN7lrQj6zFdVpFtaX3afVzZ9Iw7Fz?usp=sharing)
* Extract each archive to the `checkpoint/` directory

## Running the model in the interactive mode
* run `python3 generate.py --mode=interactive`
* type in item description after the `(>^_^)>` prompt
* if answer generation takes too long, reduce `NUMBER_OF_ANSWERS_GENERATED` constant in `generate.py`

## Evaluating model performance

* Use `generate.py` with the `--benchmark=n` parameter to generate answers for testset `n`. Examples:

Generate answers of `model-2.1` for all 10 test datasets:
```
for I in {1..10}; do python3 ./generate.py --model=2.1 --benchmark=$I; done
```
Generate answers of `model-2.1` for all 10 test datasets in parallel on two GPUs (run in separate terminal sessions):
```
for I in {1..5}; do python3 ./generate.py --model=2.1 --benchmark=$I --gpu=0; done
for I in {6..10}; do python3 ./generate.py --model=2.1 --benchmark=$I --gpu=1; done
```
* Refer to `model_benchmark.ipynb` in order to analyze quality of generated answers

## Resources
[Notebook about training GPT2 in collab](https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=LdpZQXknFNY3)

[Repository with gpt2-simple methods](https://github.com/minimaxir/gpt-2-simple/blob/master/gpt_2_simple/gpt_2.py)

[Huggingface examples](https://huggingface.co/transformers/examples.html#causal-lm-fine-tuning-on-gpt-gpt-2-masked-lm-fine-tuning-on-bert-roberta)
