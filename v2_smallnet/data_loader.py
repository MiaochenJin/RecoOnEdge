import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from params import *

def data_loader(X_train, X_test, y_train_with_energy, y_test_with_energy, train_on):
	# first extract the data that we want to use
	# TODO: also define this function for training on injection parameters
	if train_on == "zen_only":
		y_train = y_train_with_energy.T[1].T.reshape(-1, 1)
		y_test = y_test_with_energy.T[1].T.reshape(-1, 1)
	elif train_on == "azi_only":
		y_train = y_train_with_energy.T[2].T.reshape(-1, 1)
		y_test = y_test_with_energy.T[2].T.reshape(-1, 1)
	elif train_on == "e_only":
		y_train = y_train_with_energy.T[0].T.reshape(-1, 1)
		y_test = y_test_with_energy.T[0].T.reshape(-1, 1)
	elif train_on == "direction":
		train_y_x = y_train_with_energy.T[3].T.reshape(-1, 1)
		train_y_y = y_train_with_energy.T[4].T.reshape(-1, 1)
		train_y_z = y_train_with_energy.T[5].T.reshape(-1, 1)
		y_train = np.hstack([train_y_x, train_y_y, train_y_z])

		test_y_x = y_test_with_energy.T[3].T.reshape(-1, 1)
		test_y_y = y_test_with_energy.T[4].T.reshape(-1, 1)
		test_y_z = y_test_with_energy.T[5].T.reshape(-1, 1)
		y_test = np.hstack([test_y_x, test_y_y, test_y_z])
	elif train_on == "zen_dir":
		train_y_x = y_train_with_energy.T[3].T.reshape(-1, 1)
		train_y_y = y_train_with_energy.T[4].T.reshape(-1, 1)
		y_train = np.hstack([train_y_x, train_y_y])

		test_y_x = y_test_with_energy.T[3].T.reshape(-1, 1)
		test_y_y = y_test_with_energy.T[4].T.reshape(-1, 1)
		y_test = np.hstack([test_y_x, test_y_y])
	else:
		print("Invalid train_on setting provided, aborting")
		exit(1)

	# define the generators used to get datasets
	def generator():
		for i in range(len(X_train)):
			yield X_train[i], y_train[i]


	# generate from tensors does not work for large datasets, so instead use generators
	train_dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), 
												 (tf.TensorShape([SIZE_T, SIZE_H, SIZE_X, SIZE_Y]), 
												  tf.TensorShape([OUT_DIM])))
	train_dataset = train_dataset.shuffle(buffer_size = 2000)
	train_dataset = train_dataset.batch(BATCH_SIZE)

	return train_dataset, X_train, X_test, y_train, y_test


