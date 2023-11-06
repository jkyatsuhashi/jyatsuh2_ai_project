#!/usr/bin/env python3

from load_data import read_csv


def main():
    model_data = read_csv()
    print(model_data)

if __name__ == "__main__":
    main()
