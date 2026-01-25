from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_loader import get_failure_type_string


class IsolationForestDetector:
    def __init__(self, df: pd.DataFrame):
        """Initialize the detector with the full dataset.

        Args:
            df: Cleaned DataFrame from data_loader.load_dataset().
        """
        self.df = df

    def detect(
        self,
        features: list[str],
        contamination: float,
        n_estimators: int,
        max_samples: Union[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Isolation Forest on selected features.

        Args:
            features: List of column names to use as input features.
            contamination: Expected proportion of anomalies (0.01–0.20).
            n_estimators: Number of trees in the isolation forest.
            max_samples: Number of samples per tree, or "auto".

        Returns:
            Tuple of (anomaly_flags, anomaly_scores) as numpy arrays.
        """
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
        )
        X = self.df[features].values
        predictions = model.fit_predict(X)
        scores = model.decision_function(X)

        anomaly_flags = predictions == -1
        anomaly_scores = -scores  # higher = more anomalous

        return anomaly_flags, anomaly_scores

    def build_anomaly_rows(
        self,
        flags: np.ndarray,
        scores: np.ndarray,
        df: pd.DataFrame,
    ) -> list[dict]:
        """Build a list of anomaly row dicts from detection results.

        Args:
            flags: Boolean array where True indicates an anomaly.
            scores: Anomaly score array for all rows.
            df: The full dataset DataFrame.

        Returns:
            List of dicts matching the AnomalyRow schema.
        """
        rows = []
        for idx in np.where(flags)[0]:
            row = df.iloc[idx]
            rows.append(
                {
                    "row_id": int(idx),
                    "air_temp": float(row["air_temp"]),
                    "process_temp": float(row["process_temp"]),
                    "rotational_speed": float(row["rotational_speed"]),
                    "torque": float(row["torque"]),
                    "tool_wear": float(row["tool_wear"]),
                    "anomaly_score": float(scores[idx]),
                    "ground_truth_failure": int(row["machine_failure"]),
                    "failure_types": get_failure_type_string(row),
                }
            )
        return rows
