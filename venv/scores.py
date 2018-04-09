if __name__ == '__main__':

    arr = [2, 3, 6, 6, 5]

    highest = -101
    runnerup = -101

    for score in arr:
        if score > highest:
            runnerup = highest
            highest = score
        elif score > runnerup and score != highest:
            runnerup = score

    print(runnerup)
