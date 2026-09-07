from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold, cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

from datahelper import *

# Settings
input_features = ['average_breath', 'average_heart_rate', 'average_hrv',
         'deep_sleep_duration', 'light_sleep_duration', 'rem_sleep_duration']
output_feature = 'n_correct'

def training(
        columns_considered:list[str],
        training_target:str):

    # Dataset processing
    df_list = []

    for ID in range(1, 30):
        # Extract sleep data and self-report data for a single user
        data_sleep = read_id(ID, data_sleep_all())
        data_selfreport = read_id(ID, data_selfreport_all())

        # Merge both dataframes on date
        data_sleep.rename(columns={'day': 'date'}, inplace=True)
        data_all = data_sleep.merge(data_selfreport, on="date", how="left")
        data_all.rename(columns={'id_x': 'id'}, inplace=True)
        data_all.drop(columns=['id_y'], inplace=True)

        # Keep date and total_sleep_duration (for indexing), then training columns only
        df = data_all[
            ["date", "total_sleep_duration"]
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

        for col in columns_considered:
            df[col + "_average"] = df[col].shift(1).shift(1).expanding().mean()

        df = df.dropna()

        df_list.append(df)

    # Combine all participants' dataframe into one for training
    data = pd.concat(df_list, ignore_index=True)

    # Model training
    training_columns = columns_considered.copy()
    for col in columns_considered:
        training_columns.append(col + "_average")

    X = data[training_columns] # Data over the previous days
    y = data[training_target] # Sleep efficiency for the next day

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(
        rf_regressor, X, y, cv=cv,
        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
        return_estimator=True,
    )

    mae = -scores['test_neg_mean_absolute_error'].mean()
    mse = -scores['test_neg_mean_squared_error'].mean()
    r2 = scores['test_r2'].mean()

    # Check for autocorrelation
    y_pred_oof = cross_val_predict(rf_regressor, X, y, cv=cv, n_jobs=-1)

    # Cross-validated residuals
    y = pd.to_numeric(y, errors="raise").to_numpy()
    residuals = y - y_pred_oof

    # Collect feature importances from each fold
    importances =  np.vstack([
        est.feature_importances_ for est in scores["estimator"]
    ])

    # Mean and std across folds
    mean_importances = importances.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": training_columns,
        "mean_importance": mean_importances,
    }).sort_values('feature', ascending=False)

    # Dummy regressor with median strategy
    dummy = DummyRegressor(strategy="median")
    dummy.fit(X, y)

    y_pred_dummy = dummy.predict(X)

    mae_dummy = mean_absolute_error(y, y_pred_dummy)
    mse_dummy = mean_squared_error(y, y_pred_dummy)
    r2_dummy = r2_score(y, y_pred_dummy)

    return mae, mae_dummy, mse, mse_dummy, r2, r2_dummy, importance_df, residuals


if __name__ == "__main__":

    print(f"\n------------- RESULTS FOR ALL DAYS -------------")
    (mae, mae_dummy, mse, mse_dummy, r2, r2_dummy, importance, residuals) \
        = training(input_features, output_feature)

    print(f"Mean Absolute Error (vs. Dummy): {mae:.2f} / {mae_dummy:.2f}")
    print(f"Mean Squared Error (vs. Dummy): {mse:.2f} / {mse_dummy:.2f}")
    print(f"R-squared Score (vs. Dummy): {r2:.2f} / {r2_dummy:.2f}")

    print(f"Feature Importance: \n{importance}")

    lags = min(10, len(residuals) // 5)
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    print(f"\nLjung-Box test: \n{lb}")