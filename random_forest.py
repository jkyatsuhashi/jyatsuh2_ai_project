#!/usr/bin/env python3
import warnings

import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def random_forest(data):
    X = data.drop("TM_Score", axis=1).values
    y = data["TM_Score"].values.ravel()  # converts to 1D array

    # Split into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # Suppress specific warning (does not affect output)
    warnings.filterwarnings(
        "ignore",
        message="X has feature names, but RandomForestRegressor was fitted without feature names",
        category=UserWarning,
    )

    rf = RandomForestRegressor(n_estimators=2000)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    average_error = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    return rf, average_error
