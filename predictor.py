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
    return most_important


def Score_Prediction(home, away, data, nfl_data, rf):
    home_data = (
        data[nfl_data["FULL_TEAM_NAME"] == home]
        .drop("TM_Score", axis=1)
        .reset_index(drop=True)
    )
    away_data = (
        data[nfl_data["FULL_TEAM_NAME"] == away]
        .drop("TM_Score", axis=1)
        .reset_index(drop=True)
    )
    feature_names = data.drop("TM_Score", axis=1).columns
    rf.feature_names_ = feature_names
    weeks = slice(1, 16)
    home_test = pd.DataFrame(home_data[weeks].mean(axis=0)).T
    opp_columns = home_test.filter(like="Opp").columns

    home_test[opp_columns] = 0
    home_test["Opp_" + away] = 1
    home_test["Home"] = 0

    away_test = pd.DataFrame(away_data[weeks].mean(axis=0)).T
    opp_columns = away_test.filter(like="Opp").columns

    away_test[opp_columns] = 0
    away_test["Opp_" + home] = 1
    away_test["Home"] = 0

    home_test[["D_1stD", "D_Tot_Yd", "D_P_Yd", "D_R_Yd", "D_TO"]] = away_test[
        ["O_1stD", "O_Tot_Yd", "O_P_Yd", "O_R_Yd", "O_TO"]
    ]
    away_test[["D_1stD", "D_Tot_Yd", "D_P_Yd", "D_R_Yd", "D_TO"]] = home_test[
        ["O_1stD", "O_Tot_Yd", "O_P_Yd", "O_R_Yd", "O_TO"]
    ]
    X_Playoff_test = pd.concat([away_test, home_test])
    scores = rf.predict(X_Playoff_test)
    
    if scores[0] > scores[1]:
        winner = home
    else:
        winner = away
    return scores, winner
