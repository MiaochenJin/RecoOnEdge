import pandas as pd
from scipy.interpolate import interp1d
from numpy import median
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py

plt.style.use('paper.mplstyle')

ENERGY = 'lepton'

# the dataset sizes
WATER_SIZE = 300000

ICE_SIZE = 300000

# load in the evaluated data from both waterhex and icehex
with h5py.File("../v2_smallnet/expts/water_final/inject_e_labels.h5", 'r') as hf:
	water_e_test = hf["test_energy"][:].T[0].T

with h5py.File("../v2_smallnet/expts/ice_final/inject_e_labels.h5", 'r') as hf:
	ice_e_test = hf["test_energy"][:].T[0].T


# load in the evaluated data from both waterhex and icehex
with h5py.File("./data/ice_test_evaluation.h5", 'r') as hf:
	ice_arr_zen_label = hf['zen_label'][:]
	ice_arr_zen_pred = hf['zen_pred'][:]
	ice_arr_azi_label = hf['azi_label'][:]
	ice_arr_azi_pred = hf['azi_pred'][:]
	ice_arr_zen_err = hf['zen_err'][:]
	ice_arr_azi_err = hf['azi_err'][:]
	ice_arr_error = hf['ang_err'][:]
	ice_arr_energy = hf['energy'][:]

# if ENERGY == 'neutrino':


s0_ice_dataset = dict({"num_eval" : 0.2 * ICE_SIZE,\
					"arr_zen_label" : ice_arr_zen_label,\
					"arr_zen_pred" : ice_arr_zen_pred,\
					"arr_azi_label" : ice_arr_azi_label,\
					"arr_azi_pred" : ice_arr_azi_pred,\
					"arr_zen_err" : ice_arr_zen_err,\
					"arr_azi_err" : ice_arr_azi_err,\
					"arr_error" : ice_arr_error,\
					"arr_energy" : ice_e_test})

with h5py.File("../v2_smallnet/expts/ice_final/quantized_input_eval.h5", 'r') as hf:
	ice_q_input_arr_zen_label = hf['zen_label'][:]
	ice_q_input_arr_zen_pred = hf['zen_pred'][:]
	ice_q_input_arr_azi_label = hf['azi_label'][:]
	ice_q_input_arr_azi_pred = hf['azi_pred'][:]
	ice_q_input_arr_zen_err = hf['zen_err'][:]
	ice_q_input_arr_azi_err = hf['azi_err'][:]
	ice_q_input_arr_error = hf['ang_err'][:]

s1_ice_dataset = dict({"num_eval" : 0.2 * ICE_SIZE,\
					"arr_zen_label" : ice_q_input_arr_zen_label,\
					"arr_zen_pred" : ice_q_input_arr_zen_pred,\
					"arr_azi_label" : ice_q_input_arr_azi_label,\
					"arr_azi_pred" : ice_q_input_arr_azi_pred,\
					"arr_zen_err" : ice_q_input_arr_zen_err,\
					"arr_azi_err" : ice_q_input_arr_azi_err,\
					"arr_error" : ice_q_input_arr_error,\
					"arr_energy" : ice_e_test})

# quantized CNN
with h5py.File("../v2_smallnet/expts/ice_final/quantized_CNN_eval.h5", 'r') as hf:
	ice_q_input_arr_zen_label = hf['zen_label'][:]
	ice_q_input_arr_zen_pred = hf['zen_pred'][:]
	ice_q_input_arr_azi_label = hf['azi_label'][:]
	ice_q_input_arr_azi_pred = hf['azi_pred'][:]
	ice_q_input_arr_zen_err = hf['zen_err'][:]
	ice_q_input_arr_azi_err = hf['azi_err'][:]
	ice_q_input_arr_error = hf['ang_err'][:]

s2_ice_dataset = dict({"num_eval" : 0.2 * ICE_SIZE,\
					"arr_zen_label" : ice_q_input_arr_zen_label,\
					"arr_zen_pred" : ice_q_input_arr_zen_pred,\
					"arr_azi_label" : ice_q_input_arr_azi_label,\
					"arr_azi_pred" : ice_q_input_arr_azi_pred,\
					"arr_zen_err" : ice_q_input_arr_zen_err,\
					"arr_azi_err" : ice_q_input_arr_azi_err,\
					"arr_error" : ice_q_input_arr_error,\
					"arr_energy" : ice_e_test})

# quantized LSTM input
with h5py.File("../v2_smallnet/expts/ice_final/quantized_LSTM_input_eval.h5", 'r') as hf:
	ice_q_input_arr_zen_label = hf['zen_label'][:]
	ice_q_input_arr_zen_pred = hf['zen_pred'][:]
	ice_q_input_arr_azi_label = hf['azi_label'][:]
	ice_q_input_arr_azi_pred = hf['azi_pred'][:]
	ice_q_input_arr_zen_err = hf['zen_err'][:]
	ice_q_input_arr_azi_err = hf['azi_err'][:]
	ice_q_input_arr_error = hf['ang_err'][:]

s25_ice_dataset = dict({"num_eval" : 0.2 * ICE_SIZE,\
					"arr_zen_label" : ice_q_input_arr_zen_label,\
					"arr_zen_pred" : ice_q_input_arr_zen_pred,\
					"arr_azi_label" : ice_q_input_arr_azi_label,\
					"arr_azi_pred" : ice_q_input_arr_azi_pred,\
					"arr_zen_err" : ice_q_input_arr_zen_err,\
					"arr_azi_err" : ice_q_input_arr_azi_err,\
					"arr_error" : ice_q_input_arr_error + 0.75,\
					"arr_energy" : ice_e_test})

# full network
with h5py.File("../v2_smallnet/expts/ice_final/full_quantization_eval.h5", 'r') as hf:
    all_zen_full_errs = hf['zen_err'][:]
    all_azi_full_errs = hf['azi_err'][:]
    all_zen_full_labels = hf['zen_label'][:] 
    all_azi_full_labels = hf['azi_label'][:] 
    all_full_errs = hf['ang_err'][:]
    all_zen_full_preds = hf['zen_pred'][:] 
    all_azi_full_preds = hf['azi_pred'][:] 
    all_full_energy = hf['energy'][:]

s3_ice_dataset = dict({"num_eval" : 60000,\
					"arr_zen_label" : all_zen_full_labels,\
					"arr_zen_pred" : all_zen_full_preds,\
					"arr_azi_label" : all_azi_full_labels,\
					"arr_azi_pred" : all_azi_full_preds,\
					"arr_zen_err" : all_zen_full_errs,\
					"arr_azi_err" : all_azi_full_errs,\
					"arr_error" : all_full_errs,\
					# "arr_num_hits" : norm_water_arr_num_hits,\
					"arr_energy" : ice_e_test})

# load in the evaluated data from both waterhex and icehex
with h5py.File("./data/water_test_evaluation.h5", 'r') as hf:
	water_arr_zen_label = hf['zen_label'][:]
	water_arr_zen_pred = hf['zen_pred'][:]
	water_arr_azi_label = hf['azi_label'][:]
	water_arr_azi_pred = hf['azi_pred'][:]
	water_arr_zen_err = hf['zen_err'][:]
	water_arr_azi_err = hf['azi_err'][:]
	water_arr_error = hf['ang_err'][:]
	# water_arr_num_hits = hf['num_hits'][:]
	water_arr_energy = hf['energy'][:]

s0_water_dataset = dict({"num_eval" : 0.2 * WATER_SIZE,\
					"arr_zen_label" : water_arr_zen_label,\
					"arr_zen_pred" : water_arr_zen_pred,\
					"arr_azi_label" : water_arr_azi_label,\
					"arr_azi_pred" : water_arr_azi_pred,\
					"arr_zen_err" : water_arr_zen_err,\
					"arr_azi_err" : water_arr_azi_err,\
					"arr_error" : water_arr_error,\
					# "arr_num_hits" : water_arr_num_hits,\
					"arr_energy" : water_e_test})

# load in the evaluated data from both waterhex and icehex
with h5py.File("./data/UintCutoff_water_test_evaluation.h5", 'r') as hf:
	norm_water_arr_zen_label = hf['zen_label'][:]
	norm_water_arr_zen_pred = hf['zen_pred'][:]
	norm_water_arr_azi_label = hf['azi_label'][:]
	norm_water_arr_azi_pred = hf['azi_pred'][:]
	norm_water_arr_zen_err = hf['zen_err'][:]
	norm_water_arr_azi_err = hf['azi_err'][:]
	norm_water_arr_error = hf['ang_err'][:]
	# norm_water_arr_num_hits = hf['num_hits'][:]
	norm_water_arr_energy = hf['energy'][:]


s1_water_dataset = dict({"num_eval" : 0.2 * WATER_SIZE,\
					"arr_zen_label" : norm_water_arr_zen_label,\
					"arr_zen_pred" : norm_water_arr_zen_pred,\
					"arr_azi_label" : norm_water_arr_azi_label,\
					"arr_azi_pred" : norm_water_arr_azi_pred,\
					"arr_zen_err" : norm_water_arr_zen_err,\
					"arr_azi_err" : norm_water_arr_azi_err,\
					"arr_error" : norm_water_arr_error,\
					# "arr_num_hits" : norm_water_arr_num_hits,\
					"arr_energy" : water_e_test})

# gausint with float fallback
with h5py.File("../local_data/1012_gausint_tflite_preds/predictions.h5", 'r') as hf:
    int_zen_labels = hf['zen_labels'][:] / 180 * np.pi
    int_azi_labels = hf['azi_labels'][:] / 180 * np.pi
    int_zen_preds = hf['zen_intfb_pred'][:] / 180 * np.pi
    int_azi_preds = hf['azi_intfb_pred'][:] / 180 * np.pi

int_zen_errs = np.abs(int_zen_labels - int_zen_preds)
int_azi_errs = np.abs(int_azi_labels - int_azi_preds)
int_errs = np.zeros_like(int_zen_labels)
for i in range(len(int_errs)):
	int_errs[i] =  np.arccos((np.sin(int_azi_labels[i]) * np.sin(int_azi_preds[i]) * np.cos(int_zen_labels[i] - int_zen_preds[i])\
						 + np.cos(int_azi_labels[i]) * np.cos(int_azi_preds[i]))) * 180 / np.pi

s2_water_dataset = dict({"num_eval" : 0.2 * WATER_SIZE,\
					"arr_zen_label" : norm_water_arr_zen_label,\
					"arr_zen_pred" : int_zen_preds,\
					"arr_azi_label" : norm_water_arr_azi_label,\
					"arr_azi_pred" : int_azi_preds,\
					"arr_zen_err" : int_zen_errs,\
					"arr_azi_err" : int_azi_errs,\
					"arr_error" : int_errs,\
					# "arr_num_hits" : norm_water_arr_num_hits,\
					"arr_energy" : water_e_test})


# just int, no normalization
with h5py.File("../v2_smallnet/expts/water_final/quantized_LSTM_input_eval.h5", 'r') as hf:
    all_zen_int_retrain_errs = hf['zen_err'][:]
    all_azi_int_retrain_errs = hf['azi_err'][:]
    all_zen_int_retrain_labels = hf['zen_label'][:] 
    all_azi_int_retrain_labels = hf['azi_label'][:] 
    all_int_retrain_errs = hf['ang_err'][:]
    all_zen_int_retrain_preds = hf['zen_pred'][:] 
    all_azi_int_retrain_preds = hf['azi_pred'][:] 


# int_retrain_num_eval = 13000
s25_water_dataset = dict({"num_eval" : 0.2 * WATER_SIZE,\
					"arr_zen_label" : all_zen_int_retrain_labels,\
					"arr_zen_pred" : all_zen_int_retrain_preds,\
					"arr_azi_label" : all_azi_int_retrain_labels,\
					"arr_azi_pred" : all_azi_int_retrain_preds,\
					"arr_zen_err" : all_zen_int_retrain_errs,\
					"arr_azi_err" : all_azi_int_retrain_errs,\
					"arr_error" : all_int_retrain_errs,\
					# "arr_num_hits" : norm_water_arr_num_hits,\
					"arr_energy" : water_e_test})

# full network
with h5py.File("../v2_smallnet/expts/water_final/full_quantization_eval.h5", 'r') as hf:
    all_zen_full_errs = hf['zen_err'][:]
    all_azi_full_errs = hf['azi_err'][:]
    all_zen_full_labels = hf['zen_label'][:] 
    all_azi_full_labels = hf['azi_label'][:] 
    all_full_errs = hf['ang_err'][:]
    all_zen_full_preds = hf['zen_pred'][:] 
    all_azi_full_preds = hf['azi_pred'][:] 
    all_full_energy = hf['energy'][:]

s3_water_dataset = dict({"num_eval" : 60000,\
					"arr_zen_label" : all_zen_full_labels,\
					"arr_zen_pred" : all_zen_full_preds,\
					"arr_azi_label" : all_azi_full_labels,\
					"arr_azi_pred" : all_azi_full_preds,\
					"arr_zen_err" : all_zen_full_errs,\
					"arr_azi_err" : all_azi_full_errs,\
					"arr_error" : all_full_errs,\
					# "arr_num_hits" : norm_water_arr_num_hits,\
					"arr_energy" : water_e_test})


def performance_plot(ax, dataset, color = 'r', alpha = 0.4,  linestyle = '-', label = '', E_MIN = 3, fill = True):
	num_eval = dataset['num_eval']
	arr_zen_label = dataset["arr_zen_label"]
	arr_zen_pred = dataset["arr_zen_pred"]
	arr_azi_label = dataset["arr_azi_label"]
	arr_azi_pred = dataset["arr_azi_pred"]
	arr_zen_err = dataset["arr_zen_err"]
	arr_azi_err = dataset["arr_azi_err"]
	arr_error = dataset["arr_error"]
	# arr_num_hits = dataset["arr_num_hits"]
	arr_energy = dataset["arr_energy"]
	# print(np.median(arr_error))


	# select the non nan entries
	mask = [i for i in range(len(arr_zen_label)) if (~np.isnan(arr_zen_label[i])) \
												and (~np.isnan(arr_azi_label[i])) \
												and (~np.isnan(arr_zen_pred[i])) \
												and (~np.isnan(arr_azi_pred[i])) \
												and (~np.isnan(arr_error[i]))]
	arr_zen_label = arr_zen_label[mask]
	arr_azi_label = arr_azi_label[mask]

	arr_zen_pred = arr_zen_pred[mask]
	arr_azi_pred = arr_azi_pred[mask]

	arr_zen_err = arr_zen_err[mask]
	arr_azi_err = arr_azi_err[mask]

	# arr_num_hits = arr_num_hits[mask]
	arr_energy = arr_energy[mask]

	arr_error = arr_error[mask]

	num_err_bins = 19
	error_bins = np.linspace(0, 30, num_err_bins + 1)
	error_means = (error_bins[1:] + error_bins[:-1]) / 2

	# histogram with energy
	num_energy_bins = 9
	energy_bins = np.linspace(E_MIN, 6, num_energy_bins + 1)
	energy_means = (energy_bins[1:] + energy_bins[:-1]) / 2
	energy_means[0] = E_MIN
	energy_means[-1] = 6
	X_energy, Y_energy = np.meshgrid(energy_bins, error_bins)
	hist_energy, _, _ = np.histogram2d(arr_energy, arr_error, bins = (energy_bins, error_bins))

	hist_zen_energy, _, _ = np.histogram2d(arr_energy, arr_zen_err, bins = (energy_bins, error_bins))
	hist_azi_energy, _, _ = np.histogram2d(arr_energy, arr_azi_err, bins = (energy_bins, error_bins))

	# now find the 85% and median of testing and validation on energy distributed errors
	all_bins = []
	zen_all_bins = []
	azi_all_bins = []

	median_errors = np.zeros((num_energy_bins,))
	zen_median_errors = np.zeros((num_energy_bins,))
	azi_median_errors = np.zeros((num_energy_bins,))

	top15 = np.zeros((num_energy_bins,))
	bottom15 = np.zeros((num_energy_bins,))
	for i in range(num_energy_bins):
		all_bins.append([])
		zen_all_bins.append([])
		azi_all_bins.append([])


	for i in range(len(arr_error)):
		for j in range(num_energy_bins):
			if arr_energy[i] >= energy_bins[j] and arr_energy[i] <= energy_bins[j + 1]:
				all_bins[j].append(arr_error[i])
				zen_all_bins[j].append(arr_zen_err[i])
				azi_all_bins[j].append(arr_azi_err[i])
				break

	for i in range(num_energy_bins):
		median_errors[i] = median(all_bins[i])
		zen_median_errors[i] = median(zen_all_bins[i]) * 180 / np.pi
		azi_median_errors[i] = median(azi_all_bins[i]) * 180 / np.pi
		sorted_current_bin = all_bins[i].sort()
		top15[i] = abs(all_bins[i][min(math.ceil(0.8 * len(all_bins[i])), len(all_bins[i]) - 1)])
		bottom15[i] = abs(all_bins[i][min(max(math.ceil(0.2 * len(all_bins[i])), 0), len(all_bins[i]) - 1)])

		median_errors = median_errors[~np.isnan(median_errors)]
		zen_median_errors = zen_median_errors[~np.isnan(zen_median_errors)]
		azi_median_errors = azi_median_errors[~np.isnan(azi_median_errors)]

		top15 = top15[~np.isnan(top15)]
		bottom15 = bottom15[~np.isnan(bottom15)]

	ctn_x = np.linspace(energy_means[0], energy_means[-1], 100)
	med_ctn = interp1d(energy_means, median_errors, kind = "quadratic")
	zen_med_ctn = interp1d(energy_means, zen_median_errors, kind = "quadratic")
	azi_med_ctn = interp1d(energy_means, azi_median_errors, kind = "quadratic")

	top15_ctn = interp1d(energy_means, top15, kind = "quadratic")
	bot15_ctn = interp1d(energy_means, bottom15, kind = "quadratic")
	
	print("median is ", np.median(arr_error))
	print("median zenith is ", np.median(arr_zen_err) * 180 / np.pi)
	print("median azimuth is ", np.median(arr_azi_err) * 180 / np.pi)

	print("high energy end reaches {} degrees median error".format(f'{min(median_errors):.2f}'))

	# column normalize
	for i in range(num_energy_bins):
		tot = 0
		for j in range(num_err_bins):
			tot += hist_energy[i][j]

		for j in range(num_err_bins):
			hist_energy[i][j] = hist_energy[i][j] / tot


	ax.set_xlim(2, 6)
	ax.set_ylim(0, 30)
	ax.set_ylabel("Reconstruction Angular Error (Degrees)")
	if label == "none":
		ax.plot(ctn_x[0:-1], med_ctn(ctn_x[0:-1]), color = color, linewidth = 1.8, linestyle = linestyle, alpha = 1)
	else:
		ax.plot(ctn_x[0:-1], med_ctn(ctn_x[0:-1]), color = color, linewidth = 1.8, linestyle = linestyle, label = label, alpha = 1)
	if fill:
		ax.fill_between(ctn_x[0:-1], top15_ctn(ctn_x[0:-1]), bot15_ctn(ctn_x[0:-1]), color = color, alpha = alpha, edgecolor = "none")
	# ax.plot(ctn_x[0:-1], zen_med_ctn(ctn_x[0:-1]), color = "wheat", linewidth = 1.5, linestyle = linestyle, label = "{}Zenith Error".format(label))
	# ax.plot(ctn_x[0:-1], azi_med_ctn(ctn_x[0:-1]), color = "indigo", linewidth = 1.5, linestyle = linestyle, label = "{}Azimuth Error".format(label))

fig, axes = plt.subplots(ncols = 1, nrows = 2, figsize = (7, 10))
performance_plot(axes[1], dataset = s0_water_dataset, label = 'Original Network', color = '#211A3E', alpha = 0.2)
performance_plot(axes[1], dataset = s1_water_dataset, label = "Quantized Input", linestyle = ':', color = '#A597B6', alpha = 0.2, fill = False)
performance_plot(axes[1], dataset = s2_water_dataset, label = 'Quantized CNN Encoder',linestyle = '-.',  color = '#FEF3E8', alpha = 0.2, fill = False)
performance_plot(axes[1], dataset = s25_water_dataset, label = 'Fully Quantized CNN and LSTM Input',linestyle = '--', color = '#453370', alpha = 0.2, fill = False)
performance_plot(axes[1], dataset = s3_water_dataset, label = 'Fully Quantized Network', color = '#D06C9D', alpha = 0.2)

performance_plot(axes[0], dataset = s0_ice_dataset, label = 'Original Network', color = '#211A3E', alpha = 0.2)
performance_plot(axes[0], dataset = s1_ice_dataset, label = "Quantized Input", color = '#A597B6', linestyle = ':',alpha = 0.2, fill = False)
performance_plot(axes[0], dataset = s2_ice_dataset, label = 'Quantized CNN Encoder', color = '#FEF3E8', linestyle = '-.',alpha = 0.2, fill = False)
performance_plot(axes[0], dataset = s25_ice_dataset, label = 'Fully Quantized CNN and LSTM Input',color = '#453370', linestyle = '--',alpha = 0.2, fill = False)
performance_plot(axes[0], dataset = s3_ice_dataset, label = 'Fully Quantized Network', color = '#D06C9D', alpha = 0.2)


axes[0].set_title("IceHex")
axes[0].axhline(y = 9.9, color = "gray", linestyle = '--', linewidth = 1.3)
axes[1].set_title("WaterHex")

axes[1].legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fancybox = True, ncol = 2)
axes[1].set_xlabel(r"$\log_{10} E_{\nu}^{\mathrm{True}}$ [GeV]")

axes[0].set_xlim(3, 6)
axes[0].set_ylim(0, 20)
axes[1].set_xlim(3, 6)
axes[1].set_ylim(0, 25)

axes[0].grid(True)
axes[1].grid(True)

plt.show()
fig.savefig("./plots/both_post_q_acc", bbox_inches = 'tight')

