import matplotlib.pyplot as plt
import numpy as np

# Data
windows = np.array([1, 2, 3, 4, 5, 6, 7])
mse = np.array([36.02, 34.96, 33.77, 34.42, 34.58, 32.65, 31.25])
r2 = np.array([0.42, 0.44, 0.45, 0.43, 0.44, 0.47, 0.48])
missingness = np.array([0.00, 5.86, 11.61, 17.26, 22.80, 28.25, 33.59])


# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(windows, mse, marker='o', linewidth=2.5, color='crimson', label='Mean Squared Error')
ax1.set_xlabel('Results per window (day)')
ax1.set_ylabel('Mean Squared Error', color='crimson')
ax1.tick_params(axis='y', labelcolor='crimson')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(windows, r2, marker='s', linewidth=2.5, color='purple', label='R-squared Score')
ax2.set_ylabel('R-squared Score', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_ylim(0, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.title('Mean Squared Error and R-squared by Window')
plt.tight_layout()
plt.show()