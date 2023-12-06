def print_most_important(most_important):
    '''Print the most important features'''
    for index, statistic in enumerate(most_important["Var_Name"], start=1):
        print(f"{index}. {statistic}")
        if index == 10:
            return


def print_metrics(metrics):
    '''Print each of the metrics'''
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']}")
    print(f"R-Squared: {metrics['r_squared']}")
    print(f"Mean Squared Error: {metrics['mean_squared_error']}")


def print_message():
    '''Display the options menu with a rectangle around it'''
    message = "1. See Most Important Factors\n2. Predict Winner\n3. Give Score Prediction\n4. Exit"
    message = message.split("\n")
    max_length = max(len(line) for line in message)
    print("-" * 33)
    for line in message:
        print(f"| {line.ljust(max_length)} |")
    print("-" * 33)
    todo = int(input("Select an Option: "))
    print("-" * 33)
    return todo
