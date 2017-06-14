import numpy as np

# shuffle data
def ShuffleData (_ydata, _xdata1, _xdata2):

    shuffle_idx = np.arange(len(_ydata))
    np.random.shuffle(shuffle_idx)

    _ydata     = _ydata[shuffle_idx]
    _xdata1    = _xdata1[shuffle_idx]
    _xdata2    = _xdata2[shuffle_idx]

    return _ydata, _xdata1, _xdata2

def MergeData (x1, x2):
    merged = []
    for a, b in zip(x1, x2):
        merged.append(np.vstack ((a, b)))
    return np.array(merged).reshape(-1, 2, 70, 1)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)
