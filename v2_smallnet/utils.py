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

def evaluate_quant_model(interpreter_model, input_id, X, uint = True):
	interpreter = tf.lite.Interpreter(model_content=interpreter_model)
	interpreter.allocate_tensors()
	cnn_out = np.zeros((30, 128))
	for t in range(30):
		input_index = interpreter.get_input_details()[0]["index"]
		output_index = interpreter.get_output_details()[0]["index"]

		# Run predictions on every image in the "test" dataset.\
		input_tensor = tf.reshape(X[input_id][t], shape = [-1, 60, 12, 12])
		if uint:
			uint8_tensor = np.round(input_tensor).astype(np.uint8)
			interpreter.set_tensor(input_index, uint8_tensor)
		else:
			interpreter.set_tensor(input_index, input_tensor)

		# Run inference.
		interpreter.invoke()

		# Post-processing: remove batch dimension and find the digit with highest
		# probability.
		output = interpreter.tensor(output_index)
		cnn_out[t] = output()[0]
	return cnn_out

def initialize_interpreter(model_name, representative_dataset, fallback = True):
	converter_int_fb = tf.lite.TFLiteConverter.from_keras_model(model_name)
	converter_int_fb.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_int_fb.representative_dataset = representative_dataset
	converter_int_fb.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter_int_fb.inference_input_type = tf.uint8
	if not fallback:
		converter_int_fb.inference_output_type = tf.uint8

	tflite_int_fb_model = converter_int_fb.convert()

	return tflite_int_fb_model

def gen_intermediate_result(interpreter, X_train, X_test, num_train = 240000, num_test = 60000, gen_both = True):
	print("generating intermediate result")
	
	if gen_both:
		intermediate_train = np.zeros((num_train,30, 128))
		intermediate_test = np.zeros((num_test,30, 128))
		
		for i in range(num_train):
			print("\r {}/{}".format(i + 1, num_train), end = "\r", flush = True)
			intermediate_train[i] = (tf.reshape(evaluate_quant_model(interpreter,i, X = X_train), shape=[-1, 30, 128]))
		for i in range(num_test):
			print("\r {}/{}".format(i + 1, num_test), end = "\r", flush = True)
			intermediate_test[i] = (tf.reshape(evaluate_quant_model(interpreter,i, X = X_test), shape=[-1, 30, 128]))
		
		return intermediate_train, intermediate_test
	else:
		intermediate_test = np.zeros((num_test,30, 128))
		
		for i in range(num_test):
			print("\r {}/{}".format(i + 1, num_test), end = "\r", flush = True)
			intermediate_test[i] = (tf.reshape(evaluate_quant_model(interpreter,i, X = X_test), shape=[-1, 30, 128]))
		
		return intermediate_test

def replace_lstm(lstm_model, dense_model = None, version = 1):
	if version == 2:
		HIDDEN_2D = 128
		LSTM_DIM = 128
		lstm_inputs = keras.layers.Input(shape=(30, HIDDEN_2D))
		lstm_out = LSTM(LSTM_DIM)(lstm_inputs)
		fc_out = Dense(8)(lstm_out)
		outputs = Dense(1)(fc_out)

		replace_model = keras.Model(inputs = lstm_inputs, outputs = outputs)
		replace_model.layers[1].set_weights(lstm_model.layers[1].get_weights())
		replace_model.layers[2].set_weights(dense_model.layers[1].get_weights()) 
		replace_model.layers[3].set_weights(dense_model.layers[2].get_weights())    
		return replace_model
	elif version == 1:
		HIDDEN_2D = 128
		LSTM_DIM = 128
		lstm_inputs = keras.layers.Input(shape=(30, HIDDEN_2D))
		lstm_out = LSTM(LSTM_DIM)(lstm_inputs)
		outputs = Dense(1)(lstm_out)

		replace_model = keras.Model(inputs = lstm_inputs, outputs = outputs)
		replace_model.layers[1].set_weights(lstm_model.layers[1].get_weights())
		replace_model.layers[2].set_weights(lstm_model.layers[2].get_weights())  
		return replace_model
	else:
		raise ValueError("wrong version of lstm cell implementation provided")


def retrain_lstm(model_path, ckpt_path, X_train, X_test, y_train, y_test, logger = None):
	retrain_model = tf.keras.models.load_model(model_path)
	optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
	retrain_model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
	reduce_on_plat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25,
														  patience=5, min_lr=1e-10)
	

	model_checkpointing = tf.keras.callbacks.ModelCheckpoint(
		filepath=ckpt_path,
		monitor='val_loss',
		mode='min',
		save_best_only=True)

	if logger != None:
		csv_logger = tf.keras.callbacks.CSVLogger(logger, append=True)
		retrain_model.fit(X_train, y_train, batch_size = 16, epochs = 50, \
						  validation_data = (X_test, y_test), \
						 callbacks = [reduce_on_plat, model_checkpointing, csv_logger])
	else:
		retrain_model.fit(X_train, y_train, batch_size = 16, epochs = 50, \
						  validation_data = (X_test, y_test), \
						 callbacks = [reduce_on_plat, model_checkpointing])
	return retrain_model

def evaluate_lstm_model(interpreter_model, input_id, X, uint = True):
	interpreter = tf.lite.Interpreter(model_content=interpreter_model)
	interpreter.allocate_tensors()

	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

	# Run predictions on every image in the "test" dataset.\
	input_tensor = tf.reshape(X[input_id], shape = [1, 30, 128])
	if uint:
		input_tensor = np.round(input_tensor).astype(np.uint8)
	else:
		input_tensor = np.round(input_tensor).astype(np.float32)
	interpreter.set_tensor(input_index, input_tensor)

	# Run inference.
	interpreter.invoke()

	output = interpreter.tensor(output_index)
	return output()


# def quantization_performance(data, ):
# 	# initialize the model
# 	# load in the model weights





