import os
from tensorflow.keras.models import load_model

from .models.dqn_agent import dqn_mask_loss


def choose_model(path_to_models, choose=False):
    sorted_models = sorted([f for f in os.listdir(path_to_models)
                            if os.path.isfile(os.path.join(path_to_models, f)) and os.path.splitext(f)[1] == '.h5'],
                           reverse=True)
    if not sorted_models:
        return None

    if choose:
        n = len(sorted_models)
        counter = 0
        input_index = -1

        for index, model in enumerate(sorted_models):
            print("{} : {}".format(index, model))
        while counter < 5 and (input_index < 0 or input_index > n-1):
            input_index = int(input("Give the id of the model you want :"))
            counter += 1

        if counter == 5:
            print("No valid id was given. Game aborted.")
            return

        model = sorted_models[input_index]
        return load_model(os.path.join(path_to_models, model), custom_objects={'dqn_mask_loss': dqn_mask_loss})

    else:
        model = sorted_models[0]
        return load_model(os.path.join(path_to_models, model), custom_objects={'dqn_mask_loss': dqn_mask_loss})
