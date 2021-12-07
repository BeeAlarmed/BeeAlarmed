#!/usr/bin/env python3
"""! @brief This python file allows to train the BeeModel"""
##
# @file DisplayDataset.py
#
# @brief Displays random elements of the bee_dataset
#
# @section authors Author(s)
# - Created by Fabian Hickert, 2021
#

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load via TFDS
ds = tfds.load('bee_dataset/bee_dataset_300',
        batch_size=100,
        as_supervised=True,
        split="train")

# Display 
for example in ds.prefetch(tf.data.experimental.AUTOTUNE).take(1):
    for i in range(100):
        plt.imshow(example[0][i])
        plt.show()
