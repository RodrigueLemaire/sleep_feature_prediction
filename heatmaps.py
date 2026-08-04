import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_old = {
    "row": [
        "rem_sleep_duration",
        "rem_sleep_duration_mean_over_X_days",
        "light_sleep_duration",
        "light_sleep_duration_mean_over_X_days",
        "deep_sleep_duration",
        "deep_sleep_duration_mean_over_X_days",
        "average_hrv",
        "average_hrv_mean_over_X_days",
        "average_heart_rate",
        "average_heart_rate_mean_over_X_days",
        "average_breath",
        "average_breath_mean_over_X_days",
    ],
    "X=0": [0.3636, 0.0, 0.1761, 0.0, 0.1532, 0.0, 0.1031, 0.0, 0.1236, 0.0, 0.0804, 0.0],
    "X=1": [0.2855, 0.0581, 0.1601, 0.0503, 0.1061, 0.0492, 0.0522, 0.0529, 0.0468, 0.0578, 0.0349, 0.0459],
    "X=2": [0.2918, 0.0800, 0.1411, 0.0436, 0.0848, 0.0701, 0.0477, 0.0613, 0.0556, 0.0434, 0.0371, 0.0435],
    "X=3": [0.2656, 0.0788, 0.1357, 0.0534, 0.1018, 0.0863, 0.0512, 0.0580, 0.0499, 0.0420, 0.0352, 0.0419],
    "X=4": [0.2705, 0.1014, 0.1540, 0.0445, 0.0937, 0.0650, 0.0427, 0.0660, 0.0518, 0.0453, 0.0247, 0.0403],
    "X=5": [0.2527, 0.0859, 0.1608, 0.0407, 0.1055, 0.0571, 0.0471, 0.0676, 0.0590, 0.0461, 0.0358, 0.0416],
    "X=6": [0.2310, 0.1157, 0.1305, 0.0452, 0.1158, 0.0629, 0.0439, 0.0776, 0.0531, 0.0498, 0.0416, 0.0328],
}

data = {
    "row": [
        "rem_sleep_duration",
        "rem_sleep_duration_mean_over_X_days",
        "light_sleep_duration",
        "light_sleep_duration_mean_over_X_days",
        "deep_sleep_duration",
        "deep_sleep_duration_mean_over_X_days",
        "average_hrv",
        "average_hrv_mean_over_X_days",
        "average_heart_rate",
        "average_heart_rate_mean_over_X_days",
        "average_breath",
        "average_breath_mean_over_X_days",
    ],
    "X=0": [0.3357, 0.0, 0.2001, 0.0, 0.1596, 0.0, 0.1048, 0.0, 0.1150, 0.0, 0.0849, 0.0],
    "X=1": [0.2951, 0.0590, 0.1437, 0.0471, 0.1087, 0.0550, 0.0507, 0.0467, 0.0550, 0.0552, 0.0388, 0.0450],
    "X=2": [0.2771, 0.0834, 0.1368, 0.0466, 0.0977, 0.0684, 0.0449, 0.0607, 0.0538, 0.0458, 0.0380, 0.0468],
    "X=3": [0.2675, 0.0873, 0.1413, 0.0468, 0.0958, 0.0712, 0.0441, 0.0648, 0.0564, 0.0465, 0.0355, 0.0428],
    "X=4": [0.2552, 0.0996, 0.1445, 0.0495, 0.0977, 0.0644, 0.0433, 0.0693, 0.0519, 0.0470, 0.0344, 0.0433],
    "X=5": [0.2360, 0.1096, 0.1356, 0.0483, 0.1170, 0.0604, 0.0419, 0.0773, 0.0525, 0.0452, 0.0343, 0.0418],
    "X=6": [0.2357, 0.1188, 0.1235, 0.0467, 0.1098, 0.0640, 0.0439, 0.0858, 0.0533, 0.0456, 0.0349, 0.0381],
}

df = pd.DataFrame(data).set_index("row")

plt.figure(figsize=(12, 7))
sns.heatmap(df, annot=True, fmt=".4f", cmap="viridis", linewidths=0.5)
plt.title("Feature Importance Heatmap")
plt.xlabel("Window")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()