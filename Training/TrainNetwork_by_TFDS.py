#!/usr/bin/env python3
"""! @brief This python file allows to train the BeeModel"""
##
# @file TrainNetwork_by_TFDS.py 
#
# @brief This file is used to train the BeeModel via tfds dataset
#
# @section authors Author(s)
# - Created by Fabian Hickert, 2021
#

import tensorflow as tf
import tensorflow_datasets as tfds
import BeeModel

# Load via TFDS
train, val = tfds.load('bee_dataset/bee_dataset_150',
        batch_size=100,
        as_supervised=True,
        split=["train[0%:50%]", "train[50%:100%]"])

# Get the BeeModel and train it
model = BeeModel.get_bee_model(150, 75)
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
