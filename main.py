#!/usr/bin/env python3

from load_data import read_csv
from predictor import Predictor
from random_forest import random_forest


def main():
    model_data, nfl_data = read_csv()
    RF_MODEL = random_forest(model_data)
    most_important, rf = Predictor(RF_MODEL, model_data)
    print("The most important factos to consider are")
    print(most_important.head(10))


if __name__ == "__main__":
    main()
