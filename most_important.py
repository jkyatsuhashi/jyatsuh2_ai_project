def print_most_important(most_important):
    for index, statistic in enumerate(most_important["Var_Name"], start=1):
        print(f'{index}. {statistic}')
        if index == 10:
            return
