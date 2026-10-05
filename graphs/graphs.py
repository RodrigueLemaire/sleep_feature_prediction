import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Data
windows = np.array(range(0, 60))

missing_entries = [
    0.00, 0.13, 0.27, 0.42, 0.59,
    0.78, 0.97, 1.18, 1.41, 1.65,
    1.91, 2.18, 2.47, 2.77, 3.08,
    3.41, 3.75, 4.10, 4.47, 4.85,
    5.24, 5.64, 6.06, 6.49, 6.94,
    7.40, 7.87, 8.36, 8.87, 9.38,
]

mean_absolute_error = [
    3.20, 3.03, 2.95, 2.86, 2.85,
    2.82, 2.81, 2.74, 2.78, 2.72,
    2.74, 2.72, 2.64, 2.60, 2.60,
    2.63, 2.58, 2.59, 2.57, 2.58,
    2.54, 2.55, 2.56, 2.51, 2.51,
    2.53, 2.50, 2.49, 2.51, 2.52,
]

mean_squared_error = [
    15.95, 14.21, 13.78, 13.16, 13.44,
    13.00, 12.70, 12.28, 12.86, 12.26,
    12.38, 12.20, 11.75, 11.40, 11.46,
    11.60, 11.34, 11.33, 11.15, 11.35,
    11.03, 11.00, 11.04, 10.69, 10.85,
    11.09, 10.70, 10.48, 10.61, 10.68,
]

r_squared_score = [
    0.07, 0.15, 0.17, 0.21, 0.20,
    0.22, 0.24, 0.27, 0.23, 0.27,
    0.26, 0.27, 0.30, 0.32, 0.32,
    0.31, 0.33, 0.33, 0.34, 0.32,
    0.34, 0.35, 0.35, 0.37, 0.36,
    0.34, 0.37, 0.38, 0.38, 0.37,
]

missing_entries += [
    9.91, 10.46, 11.02, 11.60, 12.19,
    12.80, 13.42, 14.05, 14.70, 15.37,
    16.05, 16.74, 17.45, 18.18, 18.91,
    19.67, 20.43, 21.21, 22.00, 22.80,
    23.61, 24.43, 25.26, 26.11, 26.96,
    27.82, 28.69, 29.57, 30.45, 31.35,
]

mean_absolute_error += [
    2.51, 2.52, 2.52, 2.52, 2.51,
    2.51, 2.50, 2.51, 2.50, 2.50,
    2.49, 2.49, 2.49, 2.48, 2.49,
    2.51, 2.51, 2.49, 2.49, 2.48,
    2.49, 2.49, 2.50, 2.50, 2.52,
    2.51, 2.49, 2.50, 2.52, 2.51,
]

mean_squared_error += [
    10.63, 10.65, 10.67, 10.73, 10.64,
    10.58, 10.58, 10.65, 10.56, 10.64,
    10.47, 10.52, 10.59, 10.49, 10.53,
    10.67, 10.74, 10.59, 10.58, 10.56,
    10.71, 10.71, 10.88, 10.85, 10.98,
    10.93, 10.77, 10.80, 10.93, 10.97,
]

r_squared_score += [
    0.37, 0.37, 0.37, 0.37, 0.37,
    0.37, 0.37, 0.37, 0.38, 0.37,
    0.38, 0.38, 0.38, 0.38, 0.38,
    0.38, 0.37, 0.38, 0.38, 0.38,
    0.38, 0.38, 0.37, 0.37, 0.36,
    0.37, 0.38, 0.37, 0.37, 0.36,
]


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

    #plt.setp(ax1.xaxis.get_ticklabels(), visible=False)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=12))


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
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.set_ylim(0, len(missing_entries) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()