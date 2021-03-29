"""! @brief This python file allows to train the BeeModel"""
##
# @file train_network.py
#
# @brief This file is used to train the BeeModel which is used
#        in the BeeAlarmed project
#
# @section authors Author(s)
# - Created by Fabian Hickert, 2021
#

import BeeDataset.BeeDataset
import BeeModel
import datetime
import tensorflow as tf

import tensorflow_datasets as tfds

# Allow growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Load dataset
CFG = BeeDataset.BeeDataset.BEE_CFG_150
MODEL_SAVE_PATH = "SavedModel"

# Load via TFDS
train, val = tfds.load('bee_dataset/'+CFG.name, batch_size=100, as_supervised=True, split=["train[0%:50%]", "train[50%:100%]"])

# Get the BeeModel and train it
model = BeeModel.getBeeModel(CFG._height, CFG._width)
model.fit(
         train,
         validation_data=val,
         epochs=20,
         verbose=1,
         callbacks=[]
         )

# Save the model and show the summary
model.save(MODEL_SAVE_PATH)
model.summary()
