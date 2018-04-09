if __name__=='__main__':

    inp = [['Harry', 37.21], ['Berry', 37.21], ['Tina', 37.2], ['Akriti', 41], ['Harsh', 39]]
    lst = []

    lowest = 101

    for i in inp:
        lst.append([i[0], i[1]])
        if i[1] < lowest:
            lowest = i[1]

    lst = sorted(lst, key=lambda student: student[1])
    names = []

    lastscore = 101
    for item in lst:
        score = item[1]
        if score > lowest and not lastscore < score:
            names.append(item[0])
            lastscore = score

        #this means we finished reading all the runner-ups
        if lastscore < score:
            break

    names = sorted(names)
    for name in names:
        print(name)