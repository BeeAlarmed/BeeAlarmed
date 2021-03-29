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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

chanDim = -1
def buildVarroaBranch(inputShape):
    """! Creates the branch that detects varroa mite infestations
    """
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputShape)
    x = Conv2D(64, (4, 4), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Conv2D(32, (2, 2), padding="valid")(x)
    #x = Activation("relu")(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    #x = Conv2D(32, (2, 2), padding="valid")(x)
    #x = Activation("relu")(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid", name="varroa_output")(x)

    return x

def buildPollenBranch(inputShape):
    """! Creates the branch that detects pollen packets
    """
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputShape)
    x = Conv2D(32, (4, 4), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid", name="pollen_output")(x)

    return x

def buildWespenBranch(inputShape):
    """! Creates the branch that detects wasps
    """
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputShape)
    x = Conv2D(16, (4, 4), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(8, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1)(x)
    x = Activation("sigmoid", name="wespen_output")(x)

    return x

def buildCoolingBranch(inputShape):
    """! Created the branch that detects bees that are cooling the hive
    """
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputShape)
    x = Conv2D(64, (2, 2), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (2, 2), padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dense(1)(x)
    x = Activation("sigmoid", name="cooling_output")(x)

    return x

def getBeeModel(img_height, img_width):
    """! Creates BeeModel and returns it
    """
    inputShape = (img_height, img_width, 3)
    inputs = Input(shape=inputShape, name="input")

    pollenM = buildPollenBranch(inputs)
    varroaM = buildVarroaBranch(inputs)
    wespenM = buildWespenBranch(inputs)
    coolingM = buildCoolingBranch(inputs)

    model = Model(
        inputs=inputs,
        outputs=[varroaM, pollenM, wespenM, coolingM],
        name="beenet")

    losses = {
            "varroa_output": tf.losses.BinaryCrossentropy(),
            "pollen_output": tf.losses.BinaryCrossentropy(),
            "wespen_output": tf.losses.BinaryCrossentropy(),
            "cooling_output": tf.losses.BinaryCrossentropy()
            }
    lossWeights = {
            "varroa_output": 1.0,
            "pollen_output": 1.0,
            "wespen_output": 1.0,
            "cooling_output": 1.0
            }

    opt = Adam(lr=0.0005, decay=0.0005 / 100)

    model.compile(
         optimizer=opt,
         loss=losses,
         metrics=["accuracy"],
         loss_weights=lossWeights,
         )

    return model
