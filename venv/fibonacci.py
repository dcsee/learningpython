target = 10
count = 0
result = 0
prev_results = []
prev_results.append(0)
prev_results.append(1)

if(target == 1):
    print(prev_results[1])

if(target == 0):
    print(prev_results[0])

count = 2
while(count <= target):
    result = (prev_results[count - 1] + prev_results[count - 2])
    prev_results.append(result)
    count += 1

print(result)