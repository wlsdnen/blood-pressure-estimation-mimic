import numpy as np

# shuffle data
def ShuffleIndex (data):
    shuffled_idx = np.arange(len(data))
    np.random.shuffle(shuffled_idx)
    return shuffled_idx

def MergeData (x1, x2):
    merged = []
    for a, b in zip(x1, x2):
        merged.append(np.vstack ((a, b)))
    return np.array(merged).reshape(-1, 2, 125, 1)

def MergeChannelData (x1, x2):
    merged = []
    for a, b in zip(x1, x2):
        merged.append(np.vstack ((a, b)).T)
    return np.array(merged)


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def GetRMSE(data1, data2):

    rmse = np.sqrt(np.mean(np.square(data1 - data2)))

    return rmse