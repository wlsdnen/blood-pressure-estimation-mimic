import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

# read and save data
idx  = np.loadtxt ('data/mimic-part1-Index-001.txt', dtype=int, delimiter=',')
ydata  = np.loadtxt ('data/mimic-part1-BP-001.txt', dtype=float, delimiter=',')
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

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

idx, ydata, xdata1, xdata2 = ShuffleData (idx, ydata, xdata1, xdata2)

xdata = np.array([np.concatenate(([x], [y]), axis=1).T for x, y in zip(xdata1, xdata2)] )
ydata  = ydata[:, 0]

x_train = xdata[:25000, :].reshape(-1, 140)
y_train = ydata[:25000].reshape(-1, 1)
x_test = xdata[25000:, :].reshape(-1, 140)
y_test = ydata[25000:].reshape(-1, 1)

X = tf.placeholder(tf.float32, [None, 140])
X_signal = tf.reshape(X, [-1, 140, 1])
Y = tf.placeholder(tf.float32, [None, 1])

# Convolutional layer 1
conv1 = tf.layers.conv1d(inputs=X_signal, filters=10, kernel_size=10, padding="SAME", activation=tf.nn.relu)

print ("conv1 : ", conv1)
pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=10, padding="SAME", strides=10)
print ("pool1 : ", pool1)
flat = tf.reshape(pool1, [-1, 14 * 10])

dense = tf.layers.dense(inputs=flat, units=100, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense, units=50, activation=tf.nn.relu)

hypothesis = tf.contrib.layers.fully_connected(dense2, 1, activation_fn=None)
#
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # sum of the squares
rmse = tf.sqrt(tf.reduce_mean(tf.square(hypothesis - Y)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(30001):
     c, _, r_ = sess.run( [cost, optimizer, rmse], feed_dict={X: x_train, Y: y_train})
     if epoch % 1000 == 0:
         print (epoch, c, r_)

y_pred, loss, r = sess.run([hypothesis, cost, rmse], feed_dict={X: x_test, Y: y_test})
print ('Test result : ', loss, r)

plt.plot(y_test)
plt.plot(y_pred)

plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
