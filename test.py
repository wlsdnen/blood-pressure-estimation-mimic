import csv
import numpy as np

arr = []

file1 = open('./data/dynamic/mimic-sample-ecg.csv', 'r')
file2 = open('./data/dynamic/mimic-sample-ppg.csv', 'r')

for line1, line2 in zip(file1, file2):
    temp = []
    temp.append(np.array([float(x) for x in (line1.split(','))]))
    temp.append(np.array([float(y) for y in (line2.split(','))]))
    arr.append(temp)

arr = np.array(arr)

y_data = np.loadtxt('./data/dynamic/mimic-sample-bp.csv', delimiter=',', dtype=float)
len_data = np.loadtxt('./data/dynamic/mimic-sample-length.csv', delimiter=',', dtype=float)

print (arr.shape, y_data.shape, len_data.shape)
