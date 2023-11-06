#!/usr/bin/env python3
import numpy as np
import pandas as pd


def read_csv():
    nfl_data = pd.read_csv("2020_NFL_Game_Data.csv")
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

    model_data.loc[:, season_stats] = model_data.loc[:, season_stats] * 16

    model_data = pd.get_dummies(model_data)  # convert it to form for sklearn
    return model_data
