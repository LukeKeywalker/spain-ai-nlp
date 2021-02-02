# spain-ai-nlp
Solution of Spain AI NLP Challenge 2020

## Requirements:

* Python 3 installed
* `pip3 install gpt-2-simple`
* `pip3 install tensorflow==1.15` or ` pip3 install tensorflow-gpu==1.15` for CUDA gpus
* `pip3 install pandas`

## Running the model

* [Download one of the fine tuned models](https://drive.google.com/drive/folders/1AYZdN7lrQj6zFdVpFtaX3afVzZ9Iw7Fz?usp=sharing)
* Extract archive to `models/` directory
* run `python3 generate.py`
* type in item description after the `(>^_^)>` prompt
* if answer generation takes too long, reduce `NUMBER_OF_ANSWERS_GENERATED` constant in `generate.py`
