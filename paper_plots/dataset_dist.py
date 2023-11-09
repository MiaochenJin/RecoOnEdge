# this code is to plot the energy and zenith distribution of the input data (unweighted)
# To use for final paper paper plot, need to put all used separate files into one folder, and have a separate selected folder for trigger events

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py

HORIZONTAL = False
medium = 'ice'

plt.style.use('paper.mplstyle')

print('Loading trigger level data...')
if medium == 'water':
	df = pd.read_parquet('/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/0902_water_select_all_300787.parquet')
elif medium == 'ice':
	with h5py.File("/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/tpu_data/IceHex_7_0-0_ts=30.h5", 'r') as hf:
		y_zen = hf['y_zen'][:]
		y_e = hf['y_inj_e'][:]
	print("finished loading data with {} datapoints".format(len(y_zen)))
print('Finished')

# read in the energy and zenith
energies = []
zenith = []

for i in range(len(y_zen)):
	energies.append(np.log10(float(y_e[i])))
	zenith.append(np.cos(float(y_zen[i])))

# now make the histograms
energy_bins = np.linspace(3, 6, 20)
energy_bin_widths = energy_bins[1:] - energy_bins[:-1]
ehist, _ = np.histogram(energies, energy_bins)

zenith_bins = np.linspace(-1, 1, 15)
zenith_bin_widths = zenith_bins[1:] - zenith_bins[:-1]
zhist, _ = np.histogram(zenith, zenith_bins)

# in the next part, get th ehitogram distribution of all events at generation level and light level
directory = '/n/holyscratch01/arguelles_delgado_lab/Everyone/miaochenjin/prometheus_sim/{}_all/'.format(medium)
gen_energies = []
gen_zenith = []

light_energies = []
light_zenith = []

print('Reading generation level data...')
i = 0
for filename in os.listdir(directory):
	if filename.endswith('.parquet'):
		print("\r {}".format(i), end = "\r", flush = True)
		i += 1
		f = os.path.join(directory, filename)
		# read this file and append the events
		cur_df = pd.read_parquet(f)
		for j in range(len(cur_df['mc_truth'])):
			gen_energies.append(np.log10(float(cur_df['mc_truth'][j]['initial_state_energy'])))
			gen_zenith.append(np.cos(float(cur_df['mc_truth'][j]['initial_state_zenith'])))
			# check if there is light
			if len(cur_df['photons'][j]['t']) > 0:
				light_energies.append(np.log10(float(cur_df['mc_truth'][j]['initial_state_energy'])))
				light_zenith.append(np.cos(float(cur_df['mc_truth'][j]['initial_state_zenith'])))
print('Finished')

gen_ehist, _ = np.histogram(gen_energies, energy_bins)
gen_zhist, _ = np.histogram(gen_zenith, zenith_bins)
light_ehist, _ = np.histogram(light_energies, energy_bins)
light_zhist, _ = np.histogram(light_zenith, zenith_bins)

# print(gen_ehist - light_ehist)
# print(gen_zhist - light_zhist)
stacked_gen_ehist = [(gen_ehist[i] - light_ehist[i]) / gen_ehist[i] for i in range(len(gen_ehist))]
stacked_gen_zhist = [(gen_zhist[i] - light_zhist[i]) / gen_zhist[i] for i in range(len(gen_zhist))]
stacked_light_ehist = [(light_ehist[i] - ehist[i]) / gen_ehist[i]  for i in range(len(gen_ehist))]
stacked_light_zhist = [(light_zhist[i] - zhist[i]) / gen_zhist[i] for i in range(len(gen_zhist))]
stacked_ehist = [ehist[i] / gen_ehist[i] for i in range(len(gen_ehist))]
stacked_zhist = [zhist[i] / gen_zhist[i] for i in range(len(gen_zhist))]

print(stacked_gen_ehist)
print(stacked_light_ehist)
print(stacked_ehist)
print(stacked_gen_zhist)
print(stacked_light_zhist)
print(stacked_zhist)






# now make the plot
if HORIZONTAL:
	fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 6))
else:
	fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 9))

axes[0].hist([energy_bins[:-1], energy_bins[:-1], energy_bins[:-1]], bins = energy_bins, stacked = True, \
					weights = [stacked_ehist, stacked_light_ehist, stacked_gen_ehist], \
					color = ['#D06C9D', '#FEF3E8', '#A597B6'], \
					label = ['Trigger Level', 'Light Level', 'Generation Level'],\
					alpha = 0.7)
axes[1].hist([zenith_bins[:-1], zenith_bins[:-1], zenith_bins[:-1]], bins = zenith_bins, stacked = True, \
					weights = [stacked_zhist, stacked_light_zhist, stacked_gen_zhist], \
					color = ['#D06C9D', '#FEF3E8', '#A597B6'], \
					label = ['Trigger Level', 'Light Level', 'Generation Level'], \
					alpha = 0.7)


axes[0].set_xlabel(r"$\log_{10} E_{\nu}$ [GeV]")
axes[0].set_ylabel("Fractino of Events")
# axes[0].set_yscale('log')
axes[0].set_xlim(3, 6)
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)

# axes[0].legend()
axes[1].set_xlabel(r"$\cos(\theta_{\nu})$")
axes[1].set_ylabel("Fraction of Events")
# axes[1].set_yscale('log')
axes[1].set_xlim(-1, 1)
axes[1].legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fancybox = True, ncol = 3)
fig.tight_layout(pad = 0.1)
# plt.show()
if HORIZONTAL:
	plt.savefig('./plots/dataset_distribution_{}_HORIZONTAL'.format(medium), bbox_inches = 'tight')
else:
	plt.savefig('./plots/normed_dataset_distribution_{}'.format(medium), bbox_inches = 'tight')
