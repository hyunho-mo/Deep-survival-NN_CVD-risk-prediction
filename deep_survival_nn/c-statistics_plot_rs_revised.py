
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch

# Font config for LaTeX-style text
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'DejaVu Serif'

labels = [
    r"$\mathrm{Cox}_{\mathrm{trf}}$",
    r"$\mathrm{DL}_{\mathrm{trf}}$",
    r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{semi}}$",
    r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$",
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{auto}}$",
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{fullCT}}$",
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{heart}}$",
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$"
]

# Data
# Mean C-statistics
# Mean C-statistics
means = np.array([0.675, 0.675, 0.753, 0.757, 0.749, 0.655, 0.649, 0.749])

# 95% CI bounds
ci_lowers = np.array([0.640, 0.659, 0.717, 0.734, 0.723, 0.609, 0.620, 0.723])
ci_uppers = np.array([0.709, 0.689, 0.790, 0.780, 0.774, 0.701, 0.726, 0.774])
conf_ints = (ci_uppers - ci_lowers) / 2
best_idx = np.argmax(means)



# # Unique colors for each method
# color_palette = plt.cm.tab10.colors  # 10 distinct colors
# colors = color_palette[:len(labels)]


# Tab10 color palette
tab10 = plt.cm.tab10.colors

# Manual assignment using reordering
# Reserve indices 2 (green), 0 (blue), 3 (red)
color_mapping = {
    r"$\mathrm{DL}_{\mathrm{trf}}$": tab10[2],           # green
    r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$": tab10[0],  # blue
    r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$": tab10[3]     # red
}

# Use remaining colors for the rest (avoiding used ones: 0, 2, 3)
remaining_colors = [tab10[i] for i in range(10) if i not in [0, 2, 3]]
remaining_idx = 0

# Final ordered color list for the bar chart
colors = []
for label in labels:
    if label in color_mapping:
        colors.append(color_mapping[label])
    else:
        colors.append(remaining_colors[remaining_idx])
        remaining_idx += 1

# Mapping of methods to tab10 color indices and approximate colors (following 'labels' order)
# ------------------------------------------------------------------------------------------
# Method                                          | tab10 Index | Approx. Color
# ------------------------------------------------------------------------------------------
# r"$\mathrm{Cox}_{\mathrm{trf}}$"                | tab10[1]    | orange
# r"$\mathrm{DL}_{\mathrm{trf}}$"                 | tab10[2]    | green
# r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{semi}}$"| tab10[4]    | purple
# r"$\mathrm{Cox}_{\mathrm{trf} + \mathrm{auto}}$"| tab10[0]    | blue
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{auto}}$" | tab10[5]    | brown
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{fullCT}}$"| tab10[6]   | pink
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{heart}}$"| tab10[7]    | gray
# r"$\mathrm{DL}_{\mathrm{trf} + \mathrm{CAC}}$"  | tab10[3]    | red

# Plot
fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(range(len(labels)), means, yerr=conf_ints, capsize=8,
              color=colors, edgecolor='black', linewidth=1.2)

# Error bars (uniform linewidth)
for i, (mean, ci) in enumerate(zip(means, conf_ints)):
    ax.errorbar(i, mean, yerr=ci, capsize=8, fmt='none',
                ecolor='black', elinewidth=1.2, capthick=1.2, zorder=1)

# Remove xticks (as requested)
ax.set_xticks([])

# Annotate mean value slightly above each bar (not based on CI)
# for bar, mean in zip(bars, means):
#     # y_pos = mean + 0.015  # consistent offset above the bar
#     y_pos = mean  # consistent offset above the bar
#     ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
#             f"{mean:.3f}", ha='center', va='bottom', fontsize=10, zorder=2)

for i, (bar, mean, ci) in enumerate(zip(bars, means, conf_ints)):
    y_pos = mean + 0.005  # fixed vertical offset above the bar
    ax.text(
        bar.get_x() + bar.get_width() / 2, y_pos,
        f"{mean:.3f}",
        ha='center',
        va='bottom',
        fontsize=11,
        zorder=2,  # draw on top of error bars
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.05', alpha=0.8)
    )


# Y-axis
ax.set_ylabel("C-statistics", fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.yticks(fontsize=11)

# Legend mapping each color to method
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_elements, loc='upper left', ncol=3, fontsize=11,
          frameon=True, facecolor='white', edgecolor='none')


# Save figure
plt.tight_layout()
plt.savefig('model_comparison_ci_rs.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison_ci_rs.eps', format='eps', bbox_inches='tight')
plt.close()