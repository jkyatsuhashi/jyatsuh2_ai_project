import pandas as pd


def Most_important(rf, data):
    """Get the most important statistics in determining who will win"""
    feature_names = data.drop("TM_Score", axis=1).columns
    importance = rf.feature_importances_

    data_dictionary = {"Var_Name": feature_names, "Importance": importance}
    most_important = pd.DataFrame(data=data_dictionary)
    # Sort to get the most important features in order
    most_important.sort_values(by=["Importance"], ascending=False).reset_index(
        drop=True
    )
    return most_important


def Get_Score_Prediction(home, away, data, nfl_data, rf):
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

    # Extract the data for the home team and prepare for text
    home_test = (
        home_data[weeks].mean(axis=0).to_frame().T
    )  # Did not work without transpose
    home_opp_columns = home_test.filter(like="Opp").columns
    home_test.loc[:, home_opp_columns] = home_test.loc[:, home_opp_columns].fillna(0)
    home_test["Opp_" + away] = 1  # Set opponent
    home_test["Home"] = 0

    # Extract data for the away team and prepare for test
    away_test = away_data[weeks].mean(axis=0).to_frame().T
    away_opp_columns = away_test.filter(like="Opp").columns
    away_test.loc[:, away_opp_columns] = away_test.loc[:, away_opp_columns].fillna(0)
    away_test["Opp_" + home] = 1  # Set Opponent
    away_test["Home"] = 0

    # Swap offensive and defensive statistics to simulate a matchup between teams
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
    # Update the stats with the matchup
    home_test.update(home_defense_stats)
    away_test.update(away_defense_stats)

    # Combine the two into one for the model
    the_test = pd.concat([away_test, home_test])
    scores = rf.predict(the_test)

    # Determine the winner based on predicted scores
    winner = home if scores[0] > scores[1] else away
    return scores, winner
