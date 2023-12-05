#!/usr/bin/env python3
import numpy as np
import pandas as pd


def read_csv():
    nfl_data = pd.read_csv("nfl_game_data.csv")
    nfl_data.head(5)

    # Remove Tm, Week, Opponent Score, Result
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

    # Change to season statistics
    season_stats = [
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
    ]

    model_data.loc[:, season_stats] = (
        model_data.loc[:, season_stats] * 16
    )  # set for 16 game season
    # convert categorical variables to zeros and ones so they will work in the sklearn model
    model_data = pd.get_dummies(model_data)
    # The model_data will be ready for random forest, nfl_data preserves much of original just cleans a bit
    return model_data, nfl_data
