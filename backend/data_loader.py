import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.csv")

COLUMN_RENAME_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

DROP_COLUMNS = ["UDI", "Product ID", "Type"]

FEATURE_COLUMNS = ["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]

GROUND_TRUTH_COLUMNS = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]

FAILURE_TYPE_MAP = {
    "twf": "TWF",
    "hdf": "HDF",
    "pwf": "PWF",
    "osf": "OSF",
    "rnf": "RNF",
}


def load_dataset() -> pd.DataFrame:
    """Load the AI4I 2020 dataset, drop unused columns, and rename to snake_case.

    Returns:
        Cleaned DataFrame with renamed columns.

    Raises:
        FileNotFoundError: If ai4i2020.csv is not in the /data/ directory.
    """
    path = os.path.normpath(DATA_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place ai4i2020.csv in the /data/ directory."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def get_feature_columns() -> list[str]:
    """Return the list of sensor feature column names.

    Returns:
        List of 5 feature column name strings.
    """
    return list(FEATURE_COLUMNS)


def get_ground_truth_columns() -> list[str]:
    """Return the list of ground truth column names.

    Returns:
        List of 6 ground truth column name strings.
    """
    return list(GROUND_TRUTH_COLUMNS)


def get_failure_type_string(row: pd.Series) -> str:
    """Build a comma-separated string of active failure types for a row.

    Args:
        row: A single DataFrame row or dict-like with ground truth columns.

    Returns:
        Comma-separated failure type labels, or "None" if no failures.
    """
    active = [
        label
        for col, label in FAILURE_TYPE_MAP.items()
        if row.get(col, 0) == 1
    ]
    return ", ".join(active) if active else "None"
