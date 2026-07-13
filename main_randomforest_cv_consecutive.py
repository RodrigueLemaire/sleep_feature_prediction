from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datahelper import *

# Settings
days_window = 6
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

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    rf_regressor, X, y, cv=cv,
    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
)

mae = -scores['test_neg_mean_absolute_error'].mean()
mse = -scores['test_neg_mean_squared_error'].mean()
r2 = scores['test_r2'].mean()

# Dummy regressor with median strategy
dummy = DummyRegressor(strategy="median")
dummy.fit(X, y)

y_pred_dummy = dummy.predict(X)

mae_dummy = mean_absolute_error(y, y_pred_dummy)
mse_dummy = mean_squared_error(y, y_pred_dummy)
r2_dummy = r2_score(y, y_pred_dummy)

print(f"Missing entries: {missing_entries/29:.2f}%")

print(f"Mean Absolute Error (vs. Dummy): {mae:.2f} / {mae_dummy:.2f}")
print(f"Mean Squared Error (vs. Dummy): {mse:.2f} / {mse_dummy:.2f}")
print(f"R-squared Score (vs. Dummy): {r2:.2f} / {r2_dummy:.2f}")