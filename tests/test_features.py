import pandas as pd

from src.features import SET_A, SET_B, add_features, filter_types

POST_TXN = {"newbalanceOrig", "newbalanceDest",
            "orig_balance_error", "dest_balance_error"}


def _toy():
    return pd.DataFrame({
        "step": [1, 25, 3],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [100.0, 50.0, 10.0],
        "oldbalanceOrg": [100.0, 500.0, 0.0],
        "newbalanceOrig": [0.0, 450.0, 0.0],
        "oldbalanceDest": [0.0, 10.0, 0.0],
        "newbalanceDest": [100.0, 60.0, 0.0],
        "isFraud": [1, 0, 0],
    })


def test_set_a_has_no_post_transaction_columns():
    assert not POST_TXN & set(SET_A)
    assert POST_TXN <= set(SET_B)


def test_filter_and_drains_account():
    out = add_features(filter_types(_toy()))
    assert len(out) == 2                       # PAYMENT row removed
    assert out["drains_account"].tolist() == [1, 0]
    assert out["hour"].tolist() == [1, 1]      # step 25 -> hour 1