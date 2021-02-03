# spain-ai-nlp
Solution of Spain AI NLP Challenge 2020

## Requirements:

* Python 3 installed
* `pip3 install gpt-2-simple`
* `pip3 install tensorflow==1.15` or ` pip3 install tensorflow-gpu==1.15` for CUDA gpus
* `pip3 install pandas`
* `pip3 install alive-progress`

## Running the model in the interactive mode

* [Download one of the fine tuned models](https://drive.google.com/drive/folders/1AYZdN7lrQj6zFdVpFtaX3afVzZ9Iw7Fz?usp=sharing)
* Extract archive to `models/` directory
* run `python3 generate.py --mode=interactive`
* type in item description after the `(>^_^)>` prompt
* if answer generation takes too long, reduce `NUMBER_OF_ANSWERS_GENERATED` constant in `generate.py`

## Resources
Notebook about training GPT2 in collab
https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=LdpZQXknFNY3

Repository with gpt2-simple methods:
https://github.com/minimaxir/gpt-2-simple/blob/master/gpt_2_simple/gpt_2.py

Huggingface examples:
https://huggingface.co/transformers/examples.html#causal-lm-fine-tuning-on-gpt-gpt-2-masked-lm-fine-tuning-on-bert-roberta
