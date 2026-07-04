from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold

from datahelper import *

# Settings
days_window = 6
columns_considered = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
training_target = 'efficiency' #efficiency_tp1

# Dataset processing
df_list = []

for ID in range(1, 30):
    df = read_id(ID, data_sleep_all())

    df = df[columns_considered + ["efficiency"]]
    df["efficiency_tp1"] = df["efficiency"].copy().shift(-1)

    for days in range(1, days_window + 1):
        for col in columns_considered:
            df[col + "_tm" + str(days)] = df[col].copy().shift(days)

    df = df.dropna()

    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

# Model training
training_columns = columns_considered.copy()
for days in range(1, days_window + 1):
    for col in columns_considered:
        training_columns.append(col + "_tm" + str(days))

X = data[training_columns] # Data over the previous days
y = data[training_target] # Sleep efficiency for the next day

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    rf_regressor, X, y, cv=cv,
    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
)

mae = -scores['test_neg_mean_absolute_error'].mean()
mse = -scores['test_neg_mean_squared_error'].mean()
r2 = scores['test_r2'].mean()

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")