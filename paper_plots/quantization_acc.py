import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

# here I will write all teh accuracies:
# zenith normed original: 3.38
# zenith at various stages: 


ax1_labels = np.linspace(0, 9, 10)
ax2_labels = np.linspace(0, 9, 10)
x_ticks_positions = [1, 2, 3, 4]


# order: original accuracy (no diff), input quantization , convolution layers x 25, dense layer
Accuracies = [3.38, 4, 4.3, 4.7, 4.9, 5.0, 5.3, 5.4, 5.8, 6.0]
Differences = [0, 0.7, 0.6, 0.8, 0.7, 0.7, 0.3, 0.6, 0.7, 0.3]

fig, ax1 = plt.subplots(figsize=(8, 6))
plt.minorticks_off()

# plot the accuracies of network zenith angular reconstruction at different stages on ax1
ax1.plot(ax1_labels, Accuracies, marker = 'v', color = '#A93400', linestyle = '-', label = 'Reconstructed Zenith Error')
ax1.set_ylim(0, 8)
ax1.set_xlabel("Quantization Stages")
ax1.set_ylabel("Median Reconstruction Zenith Error [Degrees]")
ax1.legend(loc='upper left')

# plot the median angular difference with the previous layer on ax2
ax2 = ax1.twinx()
ax2.plot(ax2_labels, Differences, marker = '+', color = '#FF8C00', linestyle = '-', label = "Quantization Error")
ax2.set_ylim(0, 2)
ax2.set_ylabel("Median Quantization Zenith Error [Degrees]")

ax2.legend(loc = 'upper right')

# for now let us not reverse the y axis
# plt.gca().invert_yaxis()

fig.savefig('./plots/quantized_acc', bbox_inches='tight')