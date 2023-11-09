import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'text.usetex': True})

# Power consumption data (replace with your actual data)
power_architecture = [
    "A100 GPU on\n Lenovo Server",
    "RTX3080 GPU on\n Alienware Workstation",
    "Apple MacBook Pro\n with M1 Pro chip",
    "Google Edge TPU"
]
total_power = np.array([2400, 1000, 100, 3])
accelerator_power = np.array([400, 320, 15, 2])

# Performance data (replace with your actual data)
inference_architecture = [
    "A100 GPU on\n Lenovo Server",
    "RTX3080 GPU on\n Alienware Workstation",
    "Apple MacBook Pro\n with M1 Pro chip",
    "Google Edge TPU"
]
prediction_per_watt_total = np.array([0.03, 0.07, 0.71, 66.7])
prediction_per_watt_accelerator = np.array([0.18, 0.22, 4.76, 100])

# Combine the data
power_df = np.column_stack((power_architecture, total_power, accelerator_power))
performance_df = np.column_stack((inference_architecture, prediction_per_watt_total, prediction_per_watt_accelerator))

# Create the plot
fig, ax1 = plt.subplots(figsize=(18, 12))

# Bar plot for total power (using left y-axis)
total_power_bar = ax1.bar(np.arange(len(power_architecture)), total_power, width=0.2, color = '#A597B6', edgecolor='#A597B6', linewidth = 0.2, alpha=0.7, hatch = '.', label='Total Power (Watts)')
accelerator_power_bar = ax1.bar(np.arange(len(power_architecture)) + 0.2, accelerator_power, width=0.2, color = '#D06C9D', edgecolor='#D06C9D',linewidth = 0.3, hatch = '.',alpha=0.7, label='Accelerator Power (Watts)')

ax1.set_yscale('log')
ax1.set_xlabel('Hardware Architecture')
ax1.set_ylabel('Power (Watts)')
# ax1.set_title('Machine Learning Power Efficiency Comparison')
ax1.grid(axis='y')
x_ticks_positions = np.arange(len(power_architecture)) + 0.5
ax1.set_xticks(x_ticks_positions)
ax1.set_xticklabels(power_architecture, rotation=45, ha='right')

# Bar plot for power efficiency (using right y-axis)
ax2 = ax1.twinx()
prediction_per_watt_total_bar = ax2.bar(np.arange(len(power_architecture)) + 0.4, prediction_per_watt_total, width=0.2, color='#453370', edgecolor='#453370', alpha=0.7, hatch = '*', label='Inference Frequency per Watt (Total)')
prediction_per_watt_accelerator_bar = ax2.bar(np.arange(len(power_architecture)) + 0.6, prediction_per_watt_accelerator, width=0.2, color='#211A3E', edgecolor='#211A3E', alpha=0.7, hatch = "*", label='Inference Frequency per Watt (Accelerator)')

ax2.set_ylabel('Inference Power Efficiency (Hz/Watt)')
ax2.set_ylim(0.01, max(max(prediction_per_watt_total), max(prediction_per_watt_accelerator)) * 1.5)
ax2.set_yscale('log')

# Legends on top of the plot
bars_total = [total_power_bar, accelerator_power_bar]
bars_efficiency = [prediction_per_watt_total_bar, prediction_per_watt_accelerator_bar]
labels_total = [bar.get_label() for bar in bars_total]
labels_efficiency = [bar.get_label() for bar in bars_efficiency]
# Place the left y-axis legends on the top left
ax1.legend(bars_total, labels_total, loc='upper left', bbox_to_anchor=(0.0, 1.2), frameon=False)

# Place the right y-axis legends on the top right
ax2.legend(bars_efficiency, labels_efficiency, loc='upper right', bbox_to_anchor=(1.0, 1.2), frameon=False)



plt.tight_layout()

# Save or show the plot
plt.savefig('./plots/efficiency_barplot.png', bbox_inches='tight', dpi=600)
plt.show()
