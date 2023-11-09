from xmlrpc.client import TRANSPORT_ERROR
from model import TrainingModel, TrainingModel_v2
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
from params import *

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statistics import median
from scipy.interpolate import interp1d
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, \
	GlobalMaxPooling1D, GlobalMaxPooling2D, BatchNormalization, LayerNormalization, Layer, Add, LSTM
from tensorflow import keras

from H5data import H5Dataset
from sklearn.model_selection import train_test_split
from data_loader import data_loader

from utils import initialize_interpreter, gen_intermediate_result, evaluate_quant_model, replace_lstm, retrain_lstm, \
	evaluate_lstm_model

SAVED_Y = False

zen_model_path = "./expts/{}/train_best_zen_only.ckpt".format(EXPTNAME)
azi_model_path = "./expts/{}/train_best_azi_only.ckpt".format(EXPTNAME)
base_zen_lstm_savepath = "./expts/{}/base_zen_lstm".format(EXPTNAME)
base_azi_lstm_savepath = "./expts/{}/base_azi_lstm".format(EXPTNAME)
retrain_zen_lstm_savepath = "./expts/{}/retrain_zen_lstm.ckpt".format(EXPTNAME)
retrain_azi_lstm_savepath = "./expts/{}/retrain_azi_lstm.ckpt".format(EXPTNAME)
eval_savename = "./expts/{}/full_quantization_eval.h5".format(EXPTNAME)


num_pred = 60000

base_model_zen = TrainingModel()
base_model_azi = TrainingModel()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
base_model_zen.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
base_model_zen.load_weights(zen_model_path)
base_model_zen(np.random.normal(size=(1, 60, 12, 12, 30)))
base_zen_lstm = base_model_zen.time_slice_encoder

base_model_azi.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
base_model_azi.load_weights(azi_model_path)
base_model_azi(np.random.normal(size=(1, 60, 12, 12, 30)))
base_azi_lstm = base_model_azi.time_slice_encoder

full_zen_lstm_model = replace_lstm(base_zen_lstm)
full_azi_lstm_model = replace_lstm(base_azi_lstm)

full_zen_lstm_model.save(base_zen_lstm_savepath)
full_azi_lstm_model.save(base_azi_lstm_savepath)

# # for water, use pretrained test set!
# if MEDIUM == 'Water':
# 	full_zen_lstm_model.load_weights('./expts/water_final/retrain_int_zen_lstm.ckpt')
# 	full_azi_lstm_model.load_weights('./expts/water_final/retrain_int_azi_lstm.ckpt')

if not SAVED_Y:
	print("loading true data")
	dataset = H5Dataset(datafilename, re_normed = True)
	X_train, X_test, y_train_with_energy, y_test_with_energy = train_test_split(
		dataset.X(), dataset.Y(), test_size=0.2, random_state = 40)
	train_dataset, X_train, X_test, y_train_zen, y_test_zen \
				= data_loader(X_train, X_test, y_train_with_energy, y_test_with_energy, "zen_only")
	train_dataset, X_train, X_test, y_train_azi, y_test_azi \
				= data_loader(X_train, X_test, y_train_with_energy, y_test_with_energy, "azi_only")

	# save the y data for once
	print("saving y data...")
	with h5py.File("./expts/{}/y_labels.h5".format(EXPTNAME), 'w') as hf:
		hf.create_dataset("y_train_zen", data = y_train_zen)
		hf.create_dataset("y_test_zen", data = y_test_zen)
		hf.create_dataset("y_train_azi", data = y_train_azi)
		hf.create_dataset("y_test_azi", data = y_test_azi)
		hf.create_dataset("train_energy", data = y_train_with_energy.T[0].T.reshape(-1, 1))
		hf.create_dataset("test_energy", data = y_test_with_energy.T[0].T.reshape(-1, 1))
	print("finished saving test set values")
	train_energy = y_train_with_energy.T[0].T.reshape(-1, 1)
	test_energy = y_test_with_energy.T[0].T.reshape(-1, 1)

else:
	print("loading y data directly")
	with h5py.File("./expts/{}/y_labels.h5".format(EXPTNAME), 'r') as hf:
		y_train_zen = hf['y_train_zen'][:]
		y_test_zen = hf['y_test_zen'][:]
		y_train_azi = hf['y_train_azi'][:]
		y_test_azi = hf['y_test_azi'][:]
		train_energy = hf["train_energy"][:]
		test_energy = hf["test_energy"][:]   

print("loading intermediate data")
with h5py.File("./expts/{}/CNN_intermediate_results.h5".format(EXPTNAME), 'r') as hf:  
	int_zen_intermediate_train = hf["int_zen_intermediate_train"][:]
	int_zen_intermediate_test = hf["int_zen_intermediate_test"][:] 
	int_azi_intermediate_train = hf["int_azi_intermediate_train"][:] 
	int_azi_intermediate_test = hf["int_azi_intermediate_test"][:] 

print("retraining zenith lstm")
retrain_lstm_zen = retrain_lstm(base_zen_lstm_savepath,\
								retrain_zen_lstm_savepath, \
								int_zen_intermediate_train, int_zen_intermediate_test, 
								y_train_zen, y_test_zen, logger = "./expts/{}/retrain_lstm_zen_loss.csv".format(EXPTNAME))

print("retraining azimuth lstm")
retrain_lstm_azi = retrain_lstm(base_azi_lstm_savepath,\
								retrain_azi_lstm_savepath, \
								int_azi_intermediate_train, int_azi_intermediate_test, 
								y_train_azi, y_test_azi, logger = "./expts/{}/retrain_lstm_azi_loss.csv".format(EXPTNAME))

print("evaluating quantized LSTM input results")
zen_labels = np.zeros((num_pred,))
all_zen_retrain_pred = np.zeros((num_pred,))
azi_labels = np.zeros((num_pred,))
all_azi_retrain_pred = np.zeros((num_pred,))
all_zen_retrain_errs = np.zeros((num_pred,))
all_azi_retrain_errs = np.zeros((num_pred,))
all_err = np.zeros((num_pred,))
all_energy = np.zeros((num_pred,))

print("starting evaluation...")
for i in range(num_pred):
	if i % 100 == 0:
		print("\r {}/{}".format(i + 1, num_pred), end = "\r", flush = True)

	all_zen_retrain_pred[i] = np.arccos(retrain_lstm_zen(tf.reshape(int_zen_intermediate_test[i], shape = [-1, 30, 128])))
	all_azi_retrain_pred[i] = np.arccos(retrain_lstm_azi(tf.reshape(int_azi_intermediate_test[i], shape = [-1, 30, 128])))

	zen_labels[i] = np.arccos(y_test_zen[i])
	azi_labels[i] = np.arccos(y_test_azi[i])

	all_zen_retrain_errs[i] = np.abs(all_zen_retrain_pred[i] - zen_labels[i])
	all_azi_retrain_errs[i] = np.abs(all_azi_retrain_pred[i] - azi_labels[i])
	all_energy[i] = test_energy[i]
	
	error = np.arccos(np.sin(azi_labels[i]) * np.sin(all_azi_retrain_pred[i]) * np.cos(zen_labels[i] - all_zen_retrain_pred[i])\
							  + np.cos(azi_labels[i]) * np.cos(all_azi_retrain_pred[i])) * 180 / np.pi
	if ~np.isnan(error):
		all_err[i] = error


# save the files to h5
print("saving test set values...")
with h5py.File("./expts/{}/quantized_LSTM_input_eval.h5".format(EXPTNAME), 'w') as hf:
		hf.create_dataset("zen_label", data = zen_labels)
		hf.create_dataset("zen_pred", data = all_zen_retrain_pred)
		hf.create_dataset("azi_label", data = azi_labels)
		hf.create_dataset("azi_pred", data = all_azi_retrain_pred)
		hf.create_dataset("zen_err", data = all_zen_retrain_errs)
		hf.create_dataset("azi_err", data = all_azi_retrain_errs)
		hf.create_dataset("ang_err", data = all_err)
		hf.create_dataset("energy", data = all_energy)

print("finished saving test set values")


retrain_lstm_zen.input.set_shape((1,) + retrain_lstm_zen.input.shape[1:])
retrain_lstm_azi.input.set_shape((1,) + retrain_lstm_azi.input.shape[1:])

def zen_representative_dataset():
	for i in range(1000):
		data = np.round(tf.reshape(int_zen_intermediate_test[i], shape = [1, 30, 128])).astype(np.float32)
		yield [data]

def azi_representative_dataset():
	for i in range(1000):
		data = np.round(tf.reshape(int_azi_intermediate_test[i], shape = [1, 30, 128])).astype(np.float32)
		yield [data]

zen_converter_full = tf.lite.TFLiteConverter.from_keras_model(retrain_lstm_zen)
zen_converter_full.optimizations = [tf.lite.Optimize.DEFAULT]
zen_converter_full.representative_dataset = zen_representative_dataset
zen_converter_full.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
zen_converter_full.inference_input_type = tf.uint8
zen_tflite_full_model = zen_converter_full.convert()

azi_converter_full = tf.lite.TFLiteConverter.from_keras_model(retrain_lstm_azi)
azi_converter_full.optimizations = [tf.lite.Optimize.DEFAULT]
azi_converter_full.representative_dataset = azi_representative_dataset
azi_converter_full.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
azi_converter_full.inference_input_type = tf.uint8
azi_tflite_full_model = azi_converter_full.convert()

zen_labels = np.zeros((num_pred,))
all_zen_retrain_pred = np.zeros((num_pred,))
azi_labels = np.zeros((num_pred,))
all_azi_retrain_pred = np.zeros((num_pred,))
all_zen_retrain_errs = np.zeros((num_pred,))
all_azi_retrain_errs = np.zeros((num_pred,))
all_err = np.zeros((num_pred,))
all_energy = np.zeros((num_pred,))

for i in range(num_pred):
	if i % 100 == 0:
		print("\r {}/{}".format(i + 1, num_pred), end = "\r", flush = True)

	all_zen_retrain_pred[i] = np.arccos(evaluate_lstm_model(zen_tflite_full_model, i, X = int_zen_intermediate_test, uint = True))
	all_azi_retrain_pred[i] = np.arccos(evaluate_lstm_model(azi_tflite_full_model, i, X = int_azi_intermediate_test, uint = True))

	zen_labels[i] = np.arccos(y_test_zen[i])
	azi_labels[i] = np.arccos(y_test_azi[i])

	all_zen_retrain_errs[i] = np.abs(all_zen_retrain_pred[i] - zen_labels[i])
	all_azi_retrain_errs[i] = np.abs(all_azi_retrain_pred[i] - azi_labels[i])
	all_energy[i] = test_energy[i]
	
	error = np.arccos(np.sin(azi_labels[i]) * np.sin(all_azi_retrain_pred[i]) * np.cos(zen_labels[i] - all_zen_retrain_pred[i])\
							  + np.cos(azi_labels[i]) * np.cos(all_azi_retrain_pred[i])) * 180 / np.pi
	if ~np.isnan(error):
		all_err[i] = error


# save the files to h5
print("saving test set values...")
with h5py.File(eval_savename, 'w') as hf:
		hf.create_dataset("zen_label", data = zen_labels)
		hf.create_dataset("zen_pred", data = all_zen_retrain_pred)
		hf.create_dataset("azi_label", data = azi_labels)
		hf.create_dataset("azi_pred", data = all_azi_retrain_pred)
		hf.create_dataset("zen_err", data = all_zen_retrain_errs)
		hf.create_dataset("azi_err", data = all_azi_retrain_errs)
		hf.create_dataset("energy", data = all_energy)
		hf.create_dataset("ang_err", data = all_err)
print("finished saving test set values")


# zen_labels = np.zeros((num_pred,))
# all_zen_retrain_pred = np.zeros((num_pred,))
# azi_labels = np.zeros((num_pred,))
# all_azi_retrain_pred = np.zeros((num_pred,))
# all_zen_retrain_errs = np.zeros((num_pred,))
# all_azi_retrain_errs = np.zeros((num_pred,))
# all_err = np.zeros((num_pred,))
# all_energy = np.zeros((num_pred,))

# for i in range(num_pred):
# 	if i % 100 == 0:
# 		print("\r {}/{}".format(i + 1, num_pred), end = "\r", flush = True)

# 	all_zen_retrain_pred[i] = np.arccos(evaluate_lstm_model(zen_tflite_full_model, i, X = int_zen_intermediate_train, uint = True))
# 	all_azi_retrain_pred[i] = np.arccos(evaluate_lstm_model(azi_tflite_full_model, i, X = int_azi_intermediate_train, uint = True))

# 	zen_labels[i] = np.arccos(y_train_zen[i])
# 	azi_labels[i] = np.arccos(y_train_azi[i])

# 	all_zen_retrain_errs[i] = np.abs(all_zen_retrain_pred[i] - zen_labels[i])
# 	all_azi_retrain_errs[i] = np.abs(all_azi_retrain_pred[i] - azi_labels[i])
# 	all_energy[i] = train_energy[i]
	
# 	error = np.arccos(np.sin(azi_labels[i]) * np.sin(all_azi_retrain_pred[i]) * np.cos(zen_labels[i] - all_zen_retrain_pred[i])\
# 							  + np.cos(azi_labels[i]) * np.cos(all_azi_retrain_pred[i])) * 180 / np.pi
# 	if ~np.isnan(error):
# 		all_err[i] = error


# # save the files to h5
# print("saving test set values...")
# with h5py.File("./expts/ice_final/full_quantization_eval_training.h5", 'w') as hf:
# 		hf.create_dataset("zen_label", data = zen_labels)
# 		hf.create_dataset("zen_pred", data = all_zen_retrain_pred)
# 		hf.create_dataset("azi_label", data = azi_labels)
# 		hf.create_dataset("azi_pred", data = all_azi_retrain_pred)
# 		hf.create_dataset("zen_err", data = all_zen_retrain_errs)
# 		hf.create_dataset("azi_err", data = all_azi_retrain_errs)
# 		hf.create_dataset("energy", data = all_energy)
# 		hf.create_dataset("ang_err", data = all_err)
# print("finished saving test set values")



