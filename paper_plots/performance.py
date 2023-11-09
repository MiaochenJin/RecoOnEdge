import pandas as pd
from scipy.interpolate import interp1d
from numpy import median
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py

plt.style.use('paper.mplstyle')

# the dataset sizes
ICE_SIZE = 300000
WATER_SIZE = 300000

PLOT_HITS = False

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
	ice_arr_num_hits = hf['num_hits'][:]
	ice_arr_energy = hf['energy'][:]

if not PLOT_HITS:
	ice_arr_energy = ice_e_test

ice_dataset = dict({"num_eval" : 0.2 * ICE_SIZE,\
					"arr_zen_label" : ice_arr_zen_label,\
					"arr_zen_pred" : ice_arr_zen_pred,\
					"arr_azi_label" : ice_arr_azi_label,\
					"arr_azi_pred" : ice_arr_azi_pred,\
					"arr_zen_err" : ice_arr_zen_err,\
					"arr_azi_err" : ice_arr_azi_err,\
					"arr_error" : ice_arr_error,\
					"arr_num_hits" : ice_arr_num_hits,\
					"arr_energy" : ice_arr_energy})

# load in the evaluated data from both waterhex and icehex
with h5py.File("./data/water_test_evaluation.h5", 'r') as hf:
	water_arr_zen_label = hf['zen_label'][:]
	water_arr_zen_pred = hf['zen_pred'][:]
	water_arr_azi_label = hf['azi_label'][:]
	water_arr_azi_pred = hf['azi_pred'][:]
	water_arr_zen_err = hf['zen_err'][:]
	water_arr_azi_err = hf['azi_err'][:]
	water_arr_error = hf['ang_err'][:]
	water_arr_num_hits = hf['num_hits'][:]
	water_arr_energy = hf['energy'][:]

if not PLOT_HITS:
	water_arr_energy = water_e_test

water_dataset = dict({"num_eval" : 0.2 * WATER_SIZE,\
					"arr_zen_label" : water_arr_zen_label,\
					"arr_zen_pred" : water_arr_zen_pred,\
					"arr_azi_label" : water_arr_azi_label,\
					"arr_azi_pred" : water_arr_azi_pred,\
					"arr_zen_err" : water_arr_zen_err,\
					"arr_azi_err" : water_arr_azi_err,\
					"arr_error" : water_arr_error,\
					"arr_num_hits" : water_arr_num_hits,\
					"arr_energy" : water_arr_energy})

def performance_plot(ax, dataset = ice_dataset, label = True, plot_hits = False):
	num_eval = 2000
	arr_zen_label = dataset["arr_zen_label"]
	arr_zen_pred = dataset["arr_zen_pred"]
	arr_azi_label = dataset["arr_azi_label"]
	arr_azi_pred = dataset["arr_azi_pred"]
	arr_zen_err = dataset["arr_zen_err"]
	arr_azi_err = dataset["arr_azi_err"]
	arr_error = dataset["arr_error"]
	arr_num_hits = dataset["arr_num_hits"]
	arr_energy = dataset["arr_energy"]


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

	arr_num_hits = arr_num_hits[mask]
	arr_energy = arr_energy[mask]

	arr_error = arr_error[mask]

	num_err_bins = 19
	error_bins = np.linspace(0, 30, num_err_bins + 1)
	error_means = (error_bins[1:] + error_bins[:-1]) / 2

	# histogram with energy
	num_energy_bins = 9
	energy_min = 2 
	if not PLOT_HITS:
		energy_min = 3
	energy_bins = np.linspace(energy_min, 6, num_energy_bins + 1)

	energy_means = (energy_bins[1:] + energy_bins[:-1]) / 2
	energy_means[0] = energy_min 
	energy_means[-1] = 6.1
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
				all_bins[j].append(np.abs(arr_error[i]))
				zen_all_bins[j].append(arr_zen_err[i])
				azi_all_bins[j].append(arr_azi_err[i])
				break

	for i in range(num_energy_bins):
		median_errors[i] = median(all_bins[i])
		zen_median_errors[i] = median(zen_all_bins[i]) * 180 / np.pi
		azi_median_errors[i] = median(azi_all_bins[i]) * 180 / np.pi
		sorted_current_bin = all_bins[i].sort()
		top15[i] = abs(all_bins[i][min(math.ceil(0.85 * len(all_bins[i])), len(all_bins[i]) - 1)])
		bottom15[i] = abs(all_bins[i][min(max(math.ceil(0.15 * len(all_bins[i])), 0), len(all_bins[i]) - 1)])

		median_errors = np.abs(median_errors[~np.isnan(median_errors)])
		zen_median_errors = zen_median_errors[~np.isnan(zen_median_errors)]
		azi_median_errors = azi_median_errors[~np.isnan(azi_median_errors)]

		top15 = top15[~np.isnan(top15)]
		bottom15 = bottom15[~np.isnan(bottom15)]

	print(median_errors)
	ctn_x = np.linspace(energy_means[0], energy_means[-1], 100)
	med_ctn = interp1d(energy_means, median_errors, kind = "quadratic")
	zen_med_ctn = interp1d(energy_means, zen_median_errors, kind = "quadratic")
	azi_med_ctn = interp1d(energy_means, azi_median_errors, kind = "quadratic")

	top15_ctn = interp1d(energy_means, top15, kind = "linear")
	bot15_ctn = interp1d(energy_means, bottom15, kind = "linear")
		
	print("high energy end reaches {} degrees median error".format(f'{min(median_errors):.2f}'))

	# column normalize
	for i in range(num_energy_bins):
		tot = 0
		for j in range(num_err_bins):
			tot += hist_energy[i][j]

		for j in range(num_err_bins):
			hist_energy[i][j] = hist_energy[i][j] / tot


	# histogram with number of hits
	num_hits_bins = 9
	hits_bins = np.linspace(8, 1000, num_hits_bins + 1)
	hits_means = (hits_bins[1:] + hits_bins[:-1]) / 2
	hits_means[0] = 8 
	hits_means[-1] = 1000
	X_hits, Y_hits = np.meshgrid(hits_bins, error_bins)
	hist_hits, _, _ = np.histogram2d(arr_num_hits, arr_error, bins = (hits_bins, error_bins))

	# now find the 85% and median of testing and validation on numhits distributed errors
	hit_all_bins = []
	hit_zen_all_bins = []
	hit_azi_all_bins = []

	hit_median_errors = np.zeros((num_hits_bins,))
	hit_zen_median_errors = np.zeros((num_hits_bins,))
	hit_azi_median_errors = np.zeros((num_hits_bins,))

	hit_top15 = np.zeros((num_hits_bins,))
	hit_bottom15 = np.zeros((num_hits_bins,))
	for i in range(num_hits_bins):
		hit_all_bins.append([])
		hit_zen_all_bins.append([])
		hit_azi_all_bins.append([])


	for i in range(len(arr_error)):
		for j in range(num_hits_bins):
			if arr_num_hits[i] >= hits_bins[j] and arr_num_hits[i] <= hits_bins[j + 1]:
				hit_all_bins[j].append(arr_error[i])
				hit_zen_all_bins[j].append(arr_zen_err[i])
				hit_azi_all_bins[j].append(arr_azi_err[i])

				break

	for i in range(num_hits_bins):
		hit_median_errors[i] = median(hit_all_bins[i])
		hit_zen_median_errors[i] = median(hit_zen_all_bins[i]) * 180 / np.pi
		hit_azi_median_errors[i] = median(hit_azi_all_bins[i]) * 180 / np.pi

		hit_sorted_current_bin = hit_all_bins[i].sort()
		hit_top15[i] = abs(hit_all_bins[i][min(math.ceil(0.85 * len(hit_all_bins[i])), len(hit_all_bins[i]) - 1)])
		hit_bottom15[i] = abs(hit_all_bins[i][min(max(math.ceil(0.15 * len(hit_all_bins[i])), 0), len(hit_all_bins[i]) - 1)])

		hit_median_errors = hit_median_errors[~np.isnan(hit_median_errors)]
		hit_zen_median_errors = hit_zen_median_errors[~np.isnan(hit_zen_median_errors)]
		hit_azi_median_errors = hit_azi_median_errors[~np.isnan(hit_azi_median_errors)]

		hit_top15 = hit_top15[~np.isnan(hit_top15)]
		hit_bottom15 = hit_bottom15[~np.isnan(hit_bottom15)]

	hit_ctn_x = np.linspace(hits_means[0], hits_means[-1], 100)
	hit_med_ctn = interp1d(hits_means, hit_median_errors, kind = "quadratic")
	hit_zen_med_ctn = interp1d(hits_means, hit_zen_median_errors, kind = "quadratic")
	hit_azi_med_ctn = interp1d(hits_means, hit_azi_median_errors, kind = "quadratic")

	hit_top15_ctn = interp1d(hits_means, hit_top15, kind = "linear")
	hit_bot15_ctn = interp1d(hits_means, hit_bottom15, kind = "linear")

	# column normalize
	for i in range(num_hits_bins):
		tot = 0
		for j in range(num_err_bins):
			tot += hist_hits[i][j]
		for j in range(num_err_bins):
			hist_hits[i][j] = hist_hits[i][j] / tot

	if not plot_hits:

		ax.set_xlabel(r"$\log_{10} E_{\mu}^{\mathrm{True}}$ [GeV]")
		ax.set_xlim(2, 6)
		ax.set_ylim(0, 20)
		ax.set_ylabel("Reco Error (Degrees)")
		if label:
			ax.plot(ctn_x[0:-1], med_ctn(ctn_x[0:-1]), color = "#A597B6", linewidth = 2, linestyle = '-', label = "Angular Error")
			ax.plot(ctn_x[0:-1], zen_med_ctn(ctn_x[0:-1]), color = "#453370", linewidth = 2, linestyle = '-', label = "Zenith Error")
			ax.plot(ctn_x[0:-1], azi_med_ctn(ctn_x[0:-1]), color = "#D06C9D", linewidth = 2, linestyle = '-', label = "Azimuth Error")
		else:
			ax.plot(ctn_x[0:-1], med_ctn(ctn_x[0:-1]), color = "#A597B6", linewidth = 2, linestyle = '-')
			ax.plot(ctn_x[0:-1], zen_med_ctn(ctn_x[0:-1]), color = "#453370", linewidth = 2, linestyle = '-')
			ax.plot(ctn_x[0:-1], azi_med_ctn(ctn_x[0:-1]), color = "#D06C9D", linewidth = 2, linestyle = '-')

	else:

		ax.set_xlabel(r"$N_{\mathrm{Hits}}$")
		ax.set_xlim(8, 600)
		ax.set_ylim(0, 10)
		ax.set_ylabel("Reco Error (Degrees)")
		if label:
			ax.plot(hit_ctn_x[0:-1], hit_med_ctn(hit_ctn_x[0:-1]), color = "#A597B6", linewidth = 2, linestyle = '-', label = "Angular Error")
			ax.plot(hit_ctn_x[0:-1], hit_zen_med_ctn(hit_ctn_x[0:-1]), color = "#453370", linewidth = 2, linestyle = '-', label = "Zenith Error")
			ax.plot(hit_ctn_x[0:-1], hit_azi_med_ctn(hit_ctn_x[0:-1]), color = "#D06C9D", linewidth = 2, linestyle = '-', label = "Azimuth Error")
		else:
			ax.plot(hit_ctn_x[0:-1], hit_med_ctn(hit_ctn_x[0:-1]), color = "#A597B6", linewidth = 2, linestyle = '-')
			ax.plot(hit_ctn_x[0:-1], hit_zen_med_ctn(hit_ctn_x[0:-1]), color = "#453370", linewidth = 2, linestyle = '-')
			ax.plot(hit_ctn_x[0:-1], hit_azi_med_ctn(hit_ctn_x[0:-1]), color = "#D06C9D", linewidth = 2, linestyle = '-')

felix_y = np.array([34.24147689, 23.69394876, 15.02167538, 10.08850853, 8.91348599,\
    7.30709705, 6.20117613, 5.73722787, 5.5574623 , 4.98285052,\
    4.77118597, 4.36790919, 4.06421385, 4.10638528, 3.8371856 ,\
    3.64354274, 3.70249248, 3.74251491, 3.67219056, 3.5441045 ])

felix_x = np.log10(np.array([1.00000000e+02, 1.58489319e+02, 2.51188643e+02, 3.98107171e+02,\
    6.30957344e+02, 1.00000000e+03, 1.58489319e+03, 2.51188643e+03,\
    3.98107171e+03, 6.30957344e+03, 1.00000000e+04, 1.58489319e+04,\
    2.51188643e+04, 3.98107171e+04, 6.30957344e+04, 1.00000000e+05,\
    1.58489319e+05, 2.51188643e+05, 3.98107171e+05, 1e+06]))

if not PLOT_HITS:
	fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
	performance_plot(axes[0], dataset = ice_dataset, label = False)
	axes[0].set_title("IceHex")
	axes[0].grid(True)
	axes[1].grid(True)
	axes[0].plot(felix_x, felix_y, linestyle = '--', color = "gray", linewidth = 1.5, label = "SSCNN Benchmark")
	axes[0].set_xlim(3, 6)
	axes[0].set_ylim(0, 10)
	axes[0].set_xlabel(r"$\log_{10} E_{\nu}^{\mathrm{True}}$ [GeV]")
	axes[1].set_xlabel(r"$\log_{10} E_{\nu}^{\mathrm{True}}$ [GeV]")




	axes[0].legend(loc = 'upper right')
	performance_plot(axes[1], dataset = water_dataset)
	axes[1].set_title("WaterHex")
	axes[1].set_xlim(3, 6)
	axes[1].set_ylim(0, 10)

	axes[1].legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fancybox = True, ncol = 3)
	fig.tight_layout(pad = 0.3)
	plt.show()
	fig.savefig("./plots/performance_lepton", bbox_inches = 'tight')

	plt.close()
else:
	fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
	performance_plot(axes[0], dataset = ice_dataset, label = False)
	axes[0].set_title("IceHex")
	axes[0].grid(True)
	axes[1].grid(True)
	# axes[0].legend(loc = 'upper right')
	performance_plot(axes[1], dataset = ice_dataset, plot_hits = PLOT_HITS)
	# axes[1].set_title("WaterHex")
	axes[1].legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fancybox = True, ncol = 3)
	fig.tight_layout(pad = 0.3)
	plt.show()
	fig.savefig("./plots/performance_ice_lepton_hits", bbox_inches = 'tight')

	# plt.close()
# also plot the icehex results as a function of number of hits and muon energy


