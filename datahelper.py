import pandas as pd

# Settings
oura_sleep_cols = [
        "id",
        "day",
        "total_sleep_duration",
        "rem_sleep_duration",
        "deep_sleep_duration",
        "light_sleep_duration",
        "average_heart_rate",
        "average_hrv",
        "average_breath",
        "readiness_temperature_deviation",
        "efficiency",
        "latency",
        "restless_periods",
        "bedtime_start_delta",
        "bedtime_end_delta",
        "readiness_score"
    ]

self_report_cols = [
        "id",
        "date",
        "time",
        "n_correct",
        "selfassessment_value",
        "nback_score",
        "nback_available",
        "nback_valid",
        "selfassessment_available"
    ]

# Oura ring sleep data
def data_sleep_all() -> pd.DataFrame:
    df = pd.read_csv("oura_sleep.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)

    # keep only long and medium sleep
    df = df[df["type"].isin(["sleep", "long_sleep"])]

    columns = df.columns.to_list()

    columns_to_remove = [e for e in columns if e not in oura_sleep_cols]
    df = df.drop(columns=columns_to_remove)

    return df


# Participant information
def data_partinfo_all() -> pd.DataFrame:
    df = pd.read_csv("partinfo_survey.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)
    return df


# Self-report / N-back data
def data_selfreport_all() -> pd.DataFrame:
    df = pd.read_csv("selfreport_nback_data.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)

    columns = df.columns.to_list()

    columns_to_remove = [e for e in columns if e not in self_report_cols]
    df = df.drop(columns=columns_to_remove)

    return df

def read_id(id: int, df: pd.DataFrame) -> pd.DataFrame:
    return df[df['id'] == id]