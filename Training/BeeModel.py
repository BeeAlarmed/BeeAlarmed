"""! @brief This file contains the neural network that is used for the bee classification"""
##
# @file BeeModel.py
#
# @brief Neural network definition for the bee classification
#
# @section authors Author(s)
# - Created by Fabian Hickert, 2021
#

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

CHAN_DIM = -1
def build_varroa_branch(input_shape):
    """! Creates the branch that detects varroa mite infestations
    """
    tmp_layer= layers.experimental.preprocessing.Rescaling(1./255)(input_shape)
    tmp_layer= Conv2D(64, (4, 4), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(32, (3, 3), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(16, (3, 3), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    #tmp_layer= Conv2D(32, (2, 2), padding="valid")(tmp_layer)
    #tmp_layer= Activation("relu")(tmp_layer)
    #tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    #tmp_layer= Conv2D(32, (2, 2), padding="valid")(tmp_layer)
    #tmp_layer= Activation("relu")(tmp_layer)
    #tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Flatten()(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= Dense(1)(tmp_layer)
    tmp_layer= Activation("sigmoid", name="varroa_output")(tmp_layer)

    return tmp_layer

def build_pollen_branch(input_shape):
    """! Creates the branch that detects pollen packets
    """
    tmp_layer= layers.experimental.preprocessing.Rescaling(1./255)(input_shape)
    tmp_layer= Conv2D(32, (4, 4), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(16, (3, 3), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Flatten()(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= Dense(1)(tmp_layer)
    tmp_layer= Activation("sigmoid", name="pollen_output")(tmp_layer)

    return tmp_layer

def build_wasps_branch(input_shape):
    """! Creates the branch that detects wasps
    """
    tmp_layer= layers.experimental.preprocessing.Rescaling(1./255)(input_shape)
    tmp_layer= Conv2D(16, (4, 4), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(8, (3, 3), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Flatten()(tmp_layer)
    tmp_layer= BatchNormalization()(tmp_layer)
    tmp_layer= Dense(1)(tmp_layer)
    tmp_layer= Activation("sigmoid", name="wasps_output")(tmp_layer)

    return tmp_layer

def build_cooling_branch(input_shape):
    """! Created the branch that detects bees that are cooling the hive
    """
    tmp_layer= layers.experimental.preprocessing.Rescaling(1./255)(input_shape)
    tmp_layer= Conv2D(64, (2, 2), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(32, (3, 3), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Conv2D(16, (2, 2), padding="valid")(tmp_layer)
    tmp_layer= Activation("relu")(tmp_layer)
    tmp_layer= MaxPooling2D(pool_size=(2, 2))(tmp_layer)

    tmp_layer= Flatten()(tmp_layer)
    tmp_layer= BatchNormalization(axis=CHAN_DIM)(tmp_layer)
    tmp_layer= Dense(1)(tmp_layer)
    tmp_layer= Activation("sigmoid", name="cooling_output")(tmp_layer)

    return tmp_layer

def get_bee_model(img_height, img_width):
    """! Creates BeeModel and returns it
    """
    input_shape = (img_height, img_width, 3)
    inputs = Input(shape=input_shape, name="input")

    pollen_m = build_pollen_branch(inputs)
    varroa_m = build_varroa_branch(inputs)
    wasps_m = build_wasps_branch(inputs)
    cooling_m = build_cooling_branch(inputs)

    model = Model(
        inputs=inputs,
        outputs=[varroa_m, pollen_m, wasps_m, cooling_m],
        name="beenet")

    losses = {
            "varroa_output": tf.losses.BinaryCrossentropy(),
            "pollen_output": tf.losses.BinaryCrossentropy(),
            "wasps_output": tf.losses.BinaryCrossentropy(),
            "cooling_output": tf.losses.BinaryCrossentropy()
            }
    loss_weights = {
            "varroa_output": 1.0,
            "pollen_output": 1.0,
            "wasps_output": 1.0,
            "cooling_output": 1.0
            }

    opt = Adam(lr=0.0005, decay=0.0005 / 100)

    model.compile(
         optimizer=opt,
         loss=losses,
         metrics=["accuracy"],
         loss_weights=loss_weights,
         )

    return model
