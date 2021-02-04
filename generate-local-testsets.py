import os

import utils

CURRENT_RANDOM_STATE = 0
NUMBER_OF_RANDOM_SAMPLES = 100


def select_random_samples(sample_count, data):
    """
    Selects sample_count number of random samples from a dataframe
    :param sample_count: number of samples selected
    :param data: dataframe
    :return: dataframe with random samples
    """
    global CURRENT_RANDOM_STATE
    CURRENT_RANDOM_STATE = CURRENT_RANDOM_STATE + 1
    return data.sample(n=sample_count, random_state=CURRENT_RANDOM_STATE)


dataset = utils.load_dataset()
for i in range(100):
    random_samples = select_random_samples(NUMBER_OF_RANDOM_SAMPLES, dataset)

    try:
        os.makedirs('data/testsets')
    except OSError as e:
        pass

    utils.save_dataframe_column(
        data=random_samples,
        column_name='name',
        file_path='data/testsets/{}-names.txt'.format(i),
        header=True)
    utils.save_dataframe_column(
        data=random_samples,
        column_name='description',
        file_path='data/testsets/{}-descriptions.txt'.format(i),
        header=True
    )
