import os
from tensorflow.keras.models import load_model

from .constants import PATH_TO_MODELS
from .models.dqn_agent import DQNAgent, dqn_mask_loss


def choose_model(choose_model=False):
    path_to_models = PATH_TO_MODELS
    sorted_models = sorted([f for f in os.listdir(path_to_models) if os.isfile(os.join(path_to_models, f))], reverse=True)
    if not sorted_models:
        return None

    if choose_model:
        n = len(sorted_models)
        counter = 0
        input_index = -1

        for index, model in enumerate(sorted_models):
            print("{} : {}".format(index, model))
        while counter < 5 and (input_index < 0 or input_index > n-1):
            input_index = input("Give the id of the model you want :")
            counter += 1

        if counter == 5:
            print("No valid id was given. Game aborted.")
            return

        model = sorted_models[input_index]
        return load_model(os.join(path_to_models, model), custom_objects = {'dqn_mask_loss': dqn_mask_loss})

    else:
        model = sorted_models[0]
        return load_model(os.join(path_to_models, model), custom_objects = {'dqn_mask_loss': dqn_mask_loss})