import numpy as np

line = "-0.1904,0.119"

str_list = line.split(",")
num_list = []

for i in str_list:
    num_list.append(float(i))

a = np.round(np.average(num_list), 4)
s = np.round(np.std(num_list), 4)
print([a, s, (a / s)], end="")
