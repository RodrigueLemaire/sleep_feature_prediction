import matplotlib.pyplot as plt
import numpy as np

# Data
windows = np.array(range(0, 15))
windows_shifted = np.array(range(1, 16))

missing_entries = np.array([
    0.00, 0.13, 0.27, 0.42, 0.59,
    0.78, 0.97, 1.18, 1.41, 1.65,
    1.91, 2.18, 2.47, 2.77, 3.08
])

mean_absolute_error = np.array([
    2.85, 2.62, 2.54, 2.38, 2.31,
    2.21, 2.16, 2.14, 2.12, 2.08,
    2.09, 2.05, 2.03, 1.99, 1.99
])

mean_squared_error = np.array([
    14.74, 12.28, 11.57, 10.51, 9.94,
    9.23, 8.98, 8.86, 8.82, 8.57,
    8.62, 8.40, 8.30, 8.13, 8.08
])

r_squared_score = np.array([
    0.21, 0.31, 0.36, 0.41, 0.44,
    0.47, 0.49, 0.49, 0.50, 0.52,
    0.51, 0.53, 0.54, 0.54, 0.55
])

feature_order = [
    "average_breath",
    "average_heart_rate",
    "average_hrv",
    "deep_sleep_duration",
    "light_sleep_duration",
    "rem_sleep_duration",
    "average_breath_average",
    "average_heart_rate_average",
    "average_hrv_average",
    "deep_sleep_duration_average",
    "light_sleep_duration_average",
    "rem_sleep_duration_average",
]

feature_importance = [
    # Day 0
    [0.176, 0.235, 0.199, 0.137, 0.119, 0.131,
     0.000, 0.000, 0.000, 0.000, 0.000, 0.000],

    # Day 1
    [0.080, 0.108, 0.081, 0.068, 0.054, 0.054,
     0.099, 0.122, 0.126, 0.071, 0.053, 0.077],

    # Day 2
    [0.052, 0.093, 0.058, 0.054, 0.046, 0.053,
     0.120, 0.133, 0.164, 0.086, 0.054, 0.081],

    # Day 3
    [0.045, 0.076, 0.053, 0.048, 0.038, 0.045,
     0.121, 0.154, 0.170, 0.092, 0.053, 0.099],

    # Day 4
    [0.044, 0.070, 0.050, 0.047, 0.038, 0.043,
     0.117, 0.149, 0.184, 0.091, 0.061, 0.099],

    # Day 5
    [0.033, 0.057, 0.044, 0.041, 0.035, 0.039,
     0.125, 0.161, 0.191, 0.081, 0.084, 0.104],

    # Day 6
    [0.033, 0.048, 0.039, 0.037, 0.034, 0.037,
     0.116, 0.172, 0.193, 0.088, 0.086, 0.109],

    # Day 7
    [0.031, 0.045, 0.038, 0.036, 0.033, 0.036,
     0.121, 0.163, 0.208, 0.104, 0.079, 0.100],

    # Day 8
    [0.030, 0.044, 0.036, 0.035, 0.032, 0.037,
     0.121, 0.160, 0.203, 0.117, 0.084, 0.093],

    # Day 9
    [0.028, 0.047, 0.034, 0.034, 0.032, 0.035,
     0.112, 0.163, 0.197, 0.120, 0.090, 0.102],

    # Day 10
    [0.029, 0.043, 0.031, 0.035, 0.031, 0.034,
     0.105, 0.177, 0.194, 0.123, 0.097, 0.096],

    # Day 11
    [0.028, 0.041, 0.032, 0.034, 0.031, 0.035,
     0.096, 0.185, 0.194, 0.124, 0.102, 0.093],

    # Day 12
    [0.027, 0.038, 0.031, 0.032, 0.031, 0.035,
     0.087, 0.191, 0.191, 0.122, 0.110, 0.099],

    # Day 13
    [0.028, 0.037, 0.032, 0.032, 0.030, 0.033,
     0.091, 0.197, 0.185, 0.118, 0.109, 0.103],

    # Day 14
    [0.027, 0.036, 0.029, 0.031, 0.030, 0.033,
     0.095, 0.198, 0.179, 0.119, 0.110, 0.106],
]

if __name__ == "__main__":
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(windows, mean_squared_error, marker='o', linewidth=2.5, color='crimson', label='Mean Squared Error')
    ax1.set_xlabel('Results per window (day)')
    ax1.set_ylabel('Mean Squared Error', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.xaxis.set_ticks(windows)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(windows, r_squared_score, marker='s', linewidth=2.5, color='purple', label='R-squared Score')
    ax2.set_ylabel('R-squared Score', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.title('Mean Squared Error and R-squared by Window')
    plt.tight_layout()
    plt.show()

######

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(windows, missing_entries, color="steelblue", edgecolor="black")

    ax.set_xlabel("Window size (days)")
    ax.set_ylabel("Missingness (%)")
    ax.set_title("Missingness for different windows")

    ax.set_xticks(windows)
    ax.set_ylim(0, missing_entries.max() * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()