from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import cross_validate, KFold

from datahelper import *

# Settings
days_window = 0
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# FIXME: find a way to predict the same value each time or use mean over the whole test set
single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)

print(f"Predicted Value: {predicted_value[0]}")
print(f"Actual Value: {y_test.iloc[0]}")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
