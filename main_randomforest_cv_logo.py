import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from datahelper import *

# Settings
days_window = 100     # WARNING: Total window is equal to 1 + days_window
columns_considered = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
training_target = 'n_correct'

# Dataset processing
df_list = []
missing_entries = []

for ID in range(1, 30):
    # Extract sleep data and self-report data for a single user
    data_sleep = read_id(ID, data_sleep_all())
    data_selfreport = read_id(ID, data_selfreport_all())

    # Merge both dataframes on date
    data_sleep.rename(columns={'day': 'date'}, inplace=True)
    data_all = data_sleep.merge(data_selfreport, on="date", how="left")
    data_all.rename(columns={'id_x': 'id'}, inplace=True)
    data_all.drop(columns=['id_y'], inplace=True)

    # Keep date and total_sleep_duration (for indexing), id for LOGO, then training columns only
    df = data_all[
        ["date", "total_sleep_duration", "id"]
        + columns_considered
        + [training_target]]

    # Convert date column to a DatetimeIndex for rolling averages
    df.set_index(pd.to_datetime(df['date']), inplace=True)

    # Remove duplicate index rows by keeping entries with the most total sleep
    df = df.sort_values(by='total_sleep_duration', ascending=False)
    df = df.drop_duplicates(subset='date', keep="first")

    #df = df[~df.index.duplicated(keep='first')]

    # Ensure no days are missing, data for missing days will be NaN
    df = df.asfreq('D')

    if days_window > 0:
        for col in columns_considered:
            df[col + "_average"] = (
                df[col].shift(1).rolling(str(days_window) + 'D', min_periods=1).mean())

            # NaN count in rolling window
            missing_entries.append(
                (
                    days_window -
                    df[col].shift(1).rolling(str(days_window) + 'D', min_periods=0).count()
                ).mean()
            )

    df = df.dropna()

    df_list.append(df)

# Compute final missingness
missing_entries = np.mean(missing_entries)

# Combine all participants' dataframe into one for training
data = pd.concat(df_list, ignore_index=True)

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

print(f"Missing entries: {missing_entries:.2f}%")

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

#print(f"Feature Importance: \n{importance_df}")