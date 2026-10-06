"""Feature engineering for the fraud triage project.

Set A (deployable): only information known BEFORE the transaction completes.
Set B (forensic):   adds post-transaction balances and balance-mismatch
                    features. Used only for comparison, never for deployment.
"""
import numpy as np
import pandas as pd

MODEL_TYPES = ["TRANSFER", "CASH_OUT"]  # the only types that contain fraud

SET_A = [
    "is_transfer", "amount", "log_amount", "hour",
    "oldbalanceOrg", "oldbalanceDest",
    "drains_account", "amount_to_balance",
    "orig_old_zero", "dest_old_zero",
]

SET_B = SET_A + [
    "newbalanceOrig", "newbalanceDest",
    "orig_balance_error", "dest_balance_error",
]

TARGET = "isFraud"


def filter_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only TRANSFER and CASH_OUT rows."""
    return df[df["type"].isin(MODEL_TYPES)].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns (both sets). Pick columns with SET_A / SET_B."""
    out = df.copy()
    out["is_transfer"] = (out["type"] == "TRANSFER").astype(int)
    out["hour"] = out["step"] % 24
    out["log_amount"] = np.log1p(out["amount"])

    # --- Set A: known before the transaction completes ---
    out["drains_account"] = (
        (out["oldbalanceOrg"] > 0)
        & np.isclose(out["amount"], out["oldbalanceOrg"])
    ).astype(int)
    out["amount_to_balance"] = np.where(
        out["oldbalanceOrg"] > 0, out["amount"] / out["oldbalanceOrg"], 0.0
    ).clip(max=10)
    out["orig_old_zero"] = (out["oldbalanceOrg"] == 0).astype(int)
    out["dest_old_zero"] = (out["oldbalanceDest"] == 0).astype(int)

    # --- Set B extras: need post-transaction balances ---
    out["orig_balance_error"] = (
        out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    )
    out["dest_balance_error"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    )
    return out


def build_xy(df: pd.DataFrame, feature_set: str = "A"):
    """Return (X, y) for feature set 'A' or 'B'."""
    cols = {"A": SET_A, "B": SET_B}[feature_set]
    data = add_features(filter_types(df))
    return data[cols], data[TARGET]