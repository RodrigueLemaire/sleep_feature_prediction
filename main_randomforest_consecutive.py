from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datahelper import *

# Settings
days_window = 1
columns_considered = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
training_target = 'efficiency'

# Dataset processing
df_list = []
missing_entries = 0

for ID in range(1, 30):
    df = read_id(ID, data_sleep_all())

    df = df[columns_considered + ["efficiency"]]

    if days_window > 0:
        for col in columns_considered:
            df[col + "_average"] = df[col].shift(1).rolling(days_window).mean()

            missing_entries += df[col].shift(1).rolling(days_window, min_periods=1).count().mean()

    df = df.dropna()

    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

# Model training
training_columns = columns_considered.copy()
if days_window > 0:
    for col in columns_considered:
        training_columns.append(col + "_average")

X = data[training_columns] # Data over the previous days
y = data[training_target] # Sleep efficiency for the next day

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)

# Feature importance
importance_df = pd.DataFrame({
    'feature': training_columns,
    'importance': rf_regressor.feature_importances_
}).sort_values('feature', ascending=False)

# Dummy regressor with median strategy
dummy = DummyRegressor(strategy="median")
dummy.fit(X_train, y_train)

y_pred_dummy = dummy.predict(X_test)

mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
mse_dummy = mean_squared_error(y_test, y_pred_dummy)
r2_dummy = r2_score(y_test, y_pred_dummy)

print(f"Training dataset shape: {X_train.shape}")
print(f"Missing entries: {missing_entries/29:.2f}%")

print(f"Predicted Value: {predicted_value[0]}")
print(f"Actual Value: {y_test.iloc[0]}")

print(f"Mean Absolute Error (vs. Dummy): {mae:.2f} / {mae_dummy:.2f}")
print(f"Mean Squared Error (vs. Dummy): {mse:.2f} / {mse_dummy:.2f}")
print(f"R-squared Score (vs. Dummy): {r2:.2f} / {r2_dummy:.2f}")

print(f"Feature Importance: \n{importance_df}")