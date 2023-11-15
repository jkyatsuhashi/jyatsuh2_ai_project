import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def Predictor(rf, data):
    pd.options.display.max_rows = None
    importance = rf.feature_importances_
    feature_names = data.drop("TM_Score", axis=1).columns
    d = {"Var_Name": feature_names, "Imp": importance}
    most_important = pd.DataFrame(data=d)
    most_important.sort_values(by=["Imp"], ascending=False).reset_index(drop=True)
    return most_important, rf
