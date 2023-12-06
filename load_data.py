#!/usr/bin/env python3
import pandas as pd


def read_csv(file_path="nfl_game_data.csv"):
    """
    Read NFL game data from a CSV file, preprocess it for modeling, and return the prepared data.
    """
    # Read the CSV file
    nfl_data = pd.read_csv(file_path)

    # Just get the important categories
    model_data = nfl_data[
        [
            "FULL_TEAM_NAME",
            "Opp",
            "TM_Score",
            "O_1stD",
            "O_Tot_Yd",
            "O_P_Yd",
            "O_R_Yd",
            "O_TO",
            "D_1stD",
            "D_Tot_Yd",
            "D_P_Yd",
            "D_R_Yd",
            "D_TO",
            "Home",
            "ADA_Pred_Mean",
            "LOG_Prediction",
        ]
    ]

    # Convert categorical variables to numerical representation
    model_data = pd.get_dummies(model_data)

    return model_data, nfl_data
