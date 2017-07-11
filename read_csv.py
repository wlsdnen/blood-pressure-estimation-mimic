#import pandas as pd


#csv_test = pd.read_csv('./data/static/mimic-Part_1.csv', delimiter=',', dtype=float)


#print (csv_test.shape)


import numpy as np

data = np.loadtxt ('./data/static/mimic-Part_1-index.csv', delimiter=',', dtype=float)

np.save ('./data/static/mimic-index', data)

#data = np.load('mimic.npy')

#print (data.shape)
