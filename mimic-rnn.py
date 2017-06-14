import numpy as np
import tensorflow as tf

# read and save data
idx  = np.loadtxt ('data/mimic-part1-Index-001.txt', dtype=int, delimiter=',')
ydata  = np.loadtxt ('data/mimic-part1-BP-001.txt', dtype=int, delimiter=',')
xdata1 = np.loadtxt ('data/mimic-part1-ECG-001.txt', dtype=float, delimiter=',')
xdata2 = np.loadtxt ('data/mimic-part1-PPG-001.txt', dtype=float, delimiter=',')

# shuffle data
def ShuffleData (_indexData, _bpData, _ppgData, _ecgData):

    shuffle_idx = np.arange(len(_indexData))
    np.random.shuffle(shuffle_idx)

    _indexData  = _indexData[shuffle_idx]
    _bpData     = _bpData[shuffle_idx]
    _ppgData    = _ppgData[shuffle_idx]
    _ecgData    = _ecgData[shuffle_idx]

    return _indexData, _bpData, _ppgData, _ecgData

idx, ydata, xdata1, xdata2 = ShuffleData (idx, ydata, xdata1, xdata2)

xdata = np.array([np.concatenate(([x], [y]), axis=0).T for x, y in zip(xdata1, xdata2)] )
ydata1 = ydata[:, 0]
ydata2 = ydata[:, 1]

def MinMaxScaler (_data, _minData, _maxData):
    numerator = _data - _minData
    denominator = _maxData - _minData
    return numerator / (denominator + 1e-7)

def deMinMaxScaler (_data, _minData, _maxData):
    return (_data * (_maxData - _minData + 1e-7)) + _minData

ymin1 = np.min (ydata)
ymax1 = np.max (ydata)
ydata1 = MinMaxScaler (ydata1, ymin1, ymax1)

ymin2 = np.min (ydata)
ymax2 = np.max (ydata)
ydata2 = MinMaxScaler (ydata2, ymin2, ymax2)

# Hyper parameters
data_dim = 2
output_dim = 2
seq_length = 70

train_size = int(len(idx) * 0.7)
test_size = len(idx) - train_size

trainX, testX = np.array(xdata[0:train_size]), np.array(xdata[train_size:len(xdata)])
trainY1 , testY1  = np.array(ydata1[0:train_size]),  np.array(ydata1[train_size:len(ydata)])
trainY2 , testY2  = np.array(ydata2[0:train_size]),  np.array(ydata2[train_size:len(ydata)])
trainY  = np.concatenate(([trainY1], [trainY2]), axis=0).T
testY   = np.concatenate(([testY1], [testY2]), axis=0).T

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 2])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True)
# cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) for _ in range(5)])
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = outputs[:, -1]  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 2])
predictions = tf.placeholder(tf.float32, [None, 2])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5001):
    _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if i % 100 == 0:
        print(i, step_loss)

testPredict = sess.run(Y_pred, feed_dict={X: testX})
testPredict = deMinMaxScaler(testPredict, ymin1, ymax1)
testY       = deMinMaxScaler(testY, ymin2, ymax2)

print("RMSE", sess.run(rmse, feed_dict={targets: testY, predictions: testPredict}))

plt.plot(testY)
plt.plot(testPredict)

plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
