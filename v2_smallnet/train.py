from xmlrpc.client import TRANSPORT_ERROR
from params import *

if not BIG_MODEL:
    from model import TrainingModel
else:
    from big_model import TrainingModel
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import math
import os
from H5data import H5Dataset
from datetime import datetime
import csv
import h5py
from keras import backend as K
import sys
from tensorflow.keras.losses import CosineSimilarity
from data_loader import data_loader

f = open(savename, 'w')
f.close()

# this is to check if GPU is enabled in teh system
print("There are {} enabled GPUs in the current setup".format(len(tf.config.list_physical_devices('GPU'))))
 
# prepare data
dataset = H5Dataset(datafilename, train_on = TRAIN_ON, re_normed = IN_NORM)


print("separating and editting training/validation set")
X_train, X_test, y_train_with_energy, y_test_with_energy = train_test_split(
    dataset.X(), dataset.Y(), test_size=0.2, random_state = 40)

train_dataset, X_train, X_test, y_train, y_test \
            = data_loader(X_train, X_test, y_train_with_energy, y_test_with_energy, TRAIN_ON)


print("finished separating and editting training/validation set")

print("start training...")

# make the model
model = TrainingModel()

# lr schedule: reduce learning rate by factor if training loss not reduce for 5 epochs
reduce_on_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                                                      patience=5, min_lr=INIT_LR * 0.00001)

# save best model during training by validation loss
model_checkpointing = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# save the loss curve
csv_logger = tf.keras.callbacks.CSVLogger(savename, append=True)

# training
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)

# loss is set to mae
model.compile(optimizer=optimizer, loss='mae', metrics=["mae"])

model.fit(
    train_dataset.repeat(),
    batch_size = BATCH_SIZE,
    epochs = EPOCH,
    steps_per_epoch = int(DATA_SIZE * 0.8 / BATCH_SIZE),
    validation_data = (X_test, y_test),
    callbacks = [reduce_on_plat, model_checkpointing, csv_logger])
model.summary()


