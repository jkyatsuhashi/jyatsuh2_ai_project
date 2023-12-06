#!/usr/bin/env python3
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def random_forest(data):
    """
    Train a Random Forest Regressor on the provided data and evaluate its performance.
    """
    X = data.drop("TM_Score", axis=1).values  # all stats excluding scores
    y = data["TM_Score"].values.ravel()  # answers

    # Split into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Suppress specific warning (does not affect output)
    warnings.filterwarnings(
        "ignore",
        message="X has feature names, but RandomForestRegressor was fitted without feature names",
        category=UserWarning,
    )

    rf = RandomForestRegressor(n_estimators=2000, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # Evaluat
    metrics = {
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r_squared": r2_score(y_test, y_pred),
        "mean_squared_error": mean_squared_error(y_test, y_pred),
    }

    return rf, metrics
