#!/usr/bin/env python3

from load_data import read_csv
from predictor import Most_important, Get_Score_Prediction
from random_forest import random_forest
from print_functions import print_most_important, print_metrics, print_message


def main():
    '''
    Main driver of all the code. Gets functions from other files 
    and uses them to perform the random forest model and allow the 
    user to interact with it
    '''
    # Read data, create model, and get information
    model_data, nfl_data = read_csv()
    RF_MODEL, metrics = random_forest(model_data)
    most_important = Most_important(RF_MODEL, model_data)
    see_metrics = input("Would you like to see the performance stats of the model (y/n)? ")
    
    if see_metrics=="y":
        print_metrics(metrics)
    while 1:
        todo = print_message()
        match todo:
            case 1:
                print_most_important(most_important)
            case 2:
                try:
                    home_team = input("Enter Home Team: ")
                    away_team = input("Enter Away Team: ")
                    score, winner = Get_Score_Prediction(
                        home_team, away_team, model_data, nfl_data, RF_MODEL
                    )
                    print(f"{winner} is projected to win the game.")
                    see_score = input("Would you like to see the score (y/n)? ")
                    if see_score == "y":
                        print(f"{home_team}: {round(score[0], 1)} ")
                        print(f"{away_team}: {round(score[1], 1)} ")
                    
                except Exception as ex:
                    print("Error, make sure to enter team location and name correctly")
            case 3:
                try:
                    home_team = input("Enter Home Team: ")
                    away_team = input("Enter Away Team: ")
                    score, _ = Get_Score_Prediction(
                        home_team, away_team, model_data, nfl_data, RF_MODEL
                    )
                    print(f"{home_team}: {round(score[0], 1)} ")
                    print(f"{away_team}: {round(score[1], 1)} ")
                except Exception as ex:
                    print("Error, make sure to enter team location and name correctly")
            case 4:
                confirmation = input("Are you sure you would like to quit (y/n)? ")
                if confirmation == "y":
                    break
            case _:
                print("Please select one of the given options")
        print("-"* 33)


if __name__ == "__main__":
    main()
