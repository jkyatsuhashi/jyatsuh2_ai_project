import pandas as pd


def Most_important(rf, data):
    """Get the most important statistics in determining who will win"""
    pd.options.display.max_rows = None
    importance = rf.feature_importances_
    feature_names = data.drop("TM_Score", axis=1).columns
    data_dictionary = {"Var_Name": feature_names, "Importance": importance}
    most_important = pd.DataFrame(data=data_dictionary)
    most_important.sort_values(by=["Importance"], ascending=False).reset_index(
        drop=True
    )
    return most_important


def Score_Prediction(home, away, data, nfl_data, rf):
    """Predict scores and use this to determine winner"""

    # Used ChatGPT to figure out how to get the data from the dataframe
    # Gets the data for the given home team
    home_data = (
        data[nfl_data["FULL_TEAM_NAME"] == home]
        .drop("TM_Score", axis=1)
        .reset_index(drop=True)
    )
    # Gets the data for the given away team
    away_data = (
        data[nfl_data["FULL_TEAM_NAME"] == away]
        .drop("TM_Score", axis=1)
        .reset_index(drop=True)
    )

    # Extract feature names
    feature_names = data.drop("TM_Score", axis=1).columns
    rf.feature_names_ = feature_names

    weeks = slice(
        1, 8
    )  # So I can put in games from the last 8 weeks and see it manually
    home_test = home_data[weeks].mean(axis=0).to_frame().T
    home_opp_columns = home_test.filter(like="Opp").columns
    home_test.loc[:, home_opp_columns] = home_test.loc[:, home_opp_columns].fillna(0)
    home_test["Opp_" + away] = 1  # Check if they have played each other before
    home_test["Home"] = 0

    away_test = away_data[weeks].mean(axis=0).to_frame().T
    away_opp_columns = away_test.filter(like="Opp").columns
    away_test.loc[:, away_opp_columns] = away_test.loc[:, away_opp_columns].fillna(0)
    away_test["Opp_" + home] = 1  # Check if they have played each other before
    away_test["Home"] = 0

    # Swap offensive and defensive statistics to simulate a matchup
    home_defense_stats = away_test[
        ["O_1stD", "O_Tot_Yd", "O_P_Yd", "O_R_Yd", "O_TO"]
    ].rename(
        columns={
            "O_1stD": "D_1stD",
            "O_Tot_Yd": "D_Tot_Yd",
            "O_P_Yd": "D_P_Yd",
            "O_R_Yd": "D_R_Yd",
            "O_TO": "D_TO",
        }
    )
    away_defense_stats = home_test[
        ["O_1stD", "O_Tot_Yd", "O_P_Yd", "O_R_Yd", "O_TO"]
    ].rename(
        columns={
            "O_1stD": "D_1stD",
            "O_Tot_Yd": "D_Tot_Yd",
            "O_P_Yd": "D_P_Yd",
            "O_R_Yd": "D_R_Yd",
            "O_TO": "D_TO",
        }
    )

    home_test.update(home_defense_stats)
    away_test.update(away_defense_stats)

    # Concatenate home and away data for prediction
    the_test = pd.concat([away_test, home_test])
    scores = rf.predict(the_test)

    # Determine the winner based on predicted scores
    winner = home if scores[0] > scores[1] else away
    return scores, winner
