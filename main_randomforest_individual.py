from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from datahelper import *

# Settings
columns_considered = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
training_target = 'efficiency'

# Dataset processing
for ID in range(1, 30):
    try:
        data = read_id(ID, data_sleep_all())

        data = data[columns_considered + [training_target]]
        data["efficiency_p1"] = data["efficiency"].copy().shift(-1)
        data = data.dropna()

        # Model training
        X = data[columns_considered]
        y = data[training_target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        rf_regressor.fit(X_train, y_train)

        y_pred = rf_regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        single_data = X_test.iloc[0].values.reshape(1, -1)
        predicted_value = rf_regressor.predict(single_data)

        print(f"{ID} ======")
        print(f"Predicted Value: {predicted_value[0]}")
        print(f"Actual Value: {y_test.iloc[0]}")

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared Score: {r2:.2f}")
    except Exception as e:
        print(f"{ID}: SKIPPED WITH ERROR: {e}")