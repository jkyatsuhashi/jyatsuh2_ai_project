#!/usr/bin/env python3

from load_data import read_csv
from predictor import Predictor, Score_Prediction
from random_forest import random_forest
from most_important import print_most_important


def main():
    model_data, nfl_data = read_csv()
    rf, average_error = random_forest(model_data)
    most_important = Predictor(rf, model_data)
    if average_error > 8:
        print(f"Average Error High: {average_error}")
    while 1:
        todo = int(input("1. See Most Important Factors\n2. Predict Winner\n3. Give Score Prediction\n4. Exit\nEnter Selection: "))
        print("-"* 60)
        match todo:
            case 1:
                print_most_important(most_important)
            case 2:
                try:
                    home_team = input("Enter Home Team: ")
                    away_team = input("Enter Away Team: ")
                    _, winner = Score_Prediction(
                        home_team, away_team, model_data, nfl_data, rf
                    )
                    print(f"{winner} is projected to win the game.")
                except Exception as ex:
                    print("Error, make sure to enter team location and name correctly")
            case 3:
                try:
                    home_team = input("Enter Home Team: ")
                    away_team = input("Enter Away Team: ")
                    score, _ = Score_Prediction(
                        home_team, away_team, model_data, nfl_data, rf
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
        print("-"* 60)


if __name__ == "__main__":
    main()
