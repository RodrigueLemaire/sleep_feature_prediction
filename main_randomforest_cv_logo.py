import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from datahelper import *

# Settings
days_window = 7
columns_considered = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
training_target = 'efficiency'

# Dataset processing
df_list = []
missing_entries = 0
left_out = 1

for ID in range(1, 30):

    df = read_id(ID, data_sleep_all())
    df = df[columns_considered + ["efficiency"] + ["id"]]

    if days_window > 0:
        for col in columns_considered:
            df[col + "_average"] = df[col].shift(1).rolling(days_window).mean()

            missing_entries += df[col].shift(1).rolling(days_window, min_periods=1).count().mean()

    df = df.dropna()
    df_list.append(df)

data: pd.DataFrame = pd.concat(df_list, ignore_index=True)

# Model training
training_columns = columns_considered.copy()
if days_window > 0:
    for col in columns_considered:
        training_columns.append(col + "_average")

X = data[training_columns] # Data over the previous days
y = data[training_target] # Sleep efficiency for the next day

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

cv = LeaveOneGroupOut()
scores = cross_validate(
    rf_regressor,
    X, y,
    cv=cv,
    groups=data["id"],   # divide by participant
    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
    return_estimator=True,
)

mae = -scores['test_neg_mean_absolute_error']
mse = -scores['test_neg_mean_squared_error']
r2 = scores['test_r2']

print(f"Missing entries: {missing_entries/29:.2f}%")

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

#print(f"Feature Importance: \n{importance_df}")