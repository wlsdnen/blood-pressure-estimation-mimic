import h5py
import numpy as np

def readmat (file, varname):

    myfile = h5py.File (file, 'r')
    data = myfile[varname]

    idx = data[0][0]

    ref = data.ref

    result = []
    
    [result.append(i) for i in myfile[idx]]

    return result



def checktype (item):

    isFile    = isinstance(item, h5py.File)
    isGroup   = isinstance(item, h5py.Group)
    isDataset = isinstance(item, h5py.Dataset)

    return isFile, isGroup, isDataset


filename = 'C:/Users/jjw/Documents/MATLAB/blood-pressure-estimation/data/Part_1.mat'
var = 'Part_1'

data = readmat (filename, var)
