#!/usr/bin/env python3
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

import argparse
import tensorflow as tf
import BeeModel


# Allow growth
# pylint: disable=no-member
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true",
                    help="Use --gpu=True to train network with GPU and nothing to train on CPU.")
parser.add_argument("--local-ds", action="store_true",
                    help="Use --local-ds=True to use local BeeDataset files. Default is to use the dataset uploaded to tensorflow-datasets.")
args = parser.parse_args()

# Jetson Nano GPU
if args.gpu:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75  # limit memory allocation to avoid killing the process
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

MODEL_SAVE_PATH = "SavedModel"


# Load via TFDS
if args.local_ds:
    from BeeDataset.bee_dataset import BeeDataset
    train, val = tfds.load('bee_dataset/bee_dataset_150',
                       batch_size=11,  # 100, #changed batch to load to Jetson Nano to avoid process kill
                       as_supervised=True,
                       split=["train[0%:50%]", "train[50%:100%]"])
else:
    import tensorflow_datasets as tfds
    train, val = tfds.load('bee_dataset/bee_dataset_150',
                       batch_size=11,  # 100, #changed batch to load to Jetson Nano to avoid process kill
                       as_supervised=True,
                       split=["train[0%:50%]", "train[50%:100%]"])

if args.gpu:
    # Use GPU to train network
    with tf.device('/GPU:0'):
        # Get the BeeModel and train it
        model = BeeModel.get_bee_model(150, 75)
        model.fit(
            train,
            validation_data=val,
            epochs=20,
            verbose=1,
            callbacks=[]
        )
else:
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
