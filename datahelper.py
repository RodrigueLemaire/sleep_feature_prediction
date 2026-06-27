from functools import wraps
import logging
import pandas as pd

# Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("main.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Debug function call decorator
def call_wrapper(func):
    func_name = func.__name__
    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Logging function name and arguments
        info_to_log = \
            f"{func_name}({', '.join('% s = % r' % entry for entry in zip(arg_names, args[:len(arg_names)]))}; " \
            + f"args = {list(args[len(arg_names):])}; " \
            + f"kwargs = {kwargs})"
        logging.debug(info_to_log)

        # Call function
        result = func(*args, **kwargs)
        return result

    return wrapper


# Oura ring sleep data
@call_wrapper
def data_sleep_all() -> pd.DataFrame:
    df = pd.read_csv("oura_sleep.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)

    # keep only long and medium sleep
    df = df[df["type"].isin(["sleep", "long_sleep"])]

    columns = df.columns.to_list()

    columns_to_keep = [
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
        "bedtime_end_delta"
    ]

    columns_to_remove = [e for e in columns if e not in columns_to_keep]

    logging.debug(f"columns: {columns} \n\n columns_to_remove: {columns_to_remove}")

    df = df.drop(columns=columns_to_remove)

    return df


# Participant information
@call_wrapper
def data_partinfo_all() -> pd.DataFrame:
    df = pd.read_csv("partinfo_survey.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)
    return df


# Self-report / N-back data
@call_wrapper
def data_selfreport_all() -> pd.DataFrame:
    df = pd.read_csv("selfreport_nback_data.csv", dtype=str, index_col=0)
    df["id"] = df["id"].astype(int)

    columns = df.columns.to_list()

    columns_to_keep = [
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

    columns_to_remove = [e for e in columns if e not in columns_to_keep]

    logging.debug(f"columns: {columns} \n\n columns_to_remove: {columns_to_remove}")

    df = df.drop(columns=columns_to_remove)

    return df


@call_wrapper
def read_id(id: int, df: pd.DataFrame) -> pd.DataFrame:
    return df[df['id'] == id]