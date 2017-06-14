import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# shuffle data
def ShuffleData (_xdata, _ydata):

    shuffle_idx = np.arange(len(_ydata))
    np.random.shuffle(shuffle_idx)
    _xdata    = _xdata[shuffle_idx]
    _ydata    = _ydata[shuffle_idx]

    return _xdata, _ydata

# Load data
data = np.loadtxt ('./data/mimic-part1.csv', delimiter=',', dtype=float)
x_data = data[:, :140]
y_data = data[:, [-1]]
x_data, y_data =  ShuffleData (x_data, y_data)
print ('x_data shape :', x_data.shape)

# parameters
learning_rate = 0.001
batch_size = 8192
num_epochs = 5000

# input place holders
X = tf.placeholder(tf.float32, [None, 140])
Y = tf.placeholder(tf.float32, [None, 1])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Fully connected layers
layer1 = tf.contrib.layers.fully_connected(X, 140, activation_fn=tf.nn.relu)
drop1 = tf.nn.dropout(layer1, dropout_keep_prob)

layer2 = tf.contrib.layers.fully_connected(drop1, 140, activation_fn=tf.nn.relu)
drop2 = tf.nn.dropout(layer2, dropout_keep_prob)

layer3 = tf.contrib.layers.fully_connected(drop2, 140, activation_fn=tf.nn.relu)
drop3 = tf.nn.dropout(layer3, dropout_keep_prob)

layer4 = tf.contrib.layers.fully_connected(drop3, 140, activation_fn=tf.nn.relu)
drop4 = tf.nn.dropout(layer4, dropout_keep_prob)

# Output layer
predictions = tf.contrib.layers.fully_connected(drop4, 1, activation_fn=None)

# define cost/loss & optimizer
cost = tf.sqrt(tf.reduce_mean(tf.square(predictions - Y)))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_size  = int(len(y_data) * 0.7)
test_size   = len(y_data) - train_size

trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])

for epoch in range(num_epochs+1):

    avg_cost    = 0
    total_batch = int((len(trainX)-1)/batch_size) + 1

    for iteration in range(total_batch):

        start_index = iteration * batch_size
        end_index = min((iteration + 1) * batch_size, train_size)
        batch_x = trainX[start_index:end_index]
        batch_y = trainY[start_index:end_index]
        # A single training step
        _, loss = sess.run ([train, cost], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 0.5})
        avg_cost += loss / total_batch

    if epoch % 100 == 0:
        print('Epoch:', '%03d' % epoch, 'rmse =', avg_cost)

avg_cost    = 0
total_batch = int((len(testX)-1)/batch_size) + 1

predict = np.array([])

for iteration in range(total_batch):

    start_index = iteration * batch_size
    end_index = min((iteration + 1) * batch_size, train_size)
    batch_x = testX[start_index:end_index]
    batch_y = testY[start_index:end_index]

    a, pred = sess.run ([cost, predictions], feed_dict={X: batch_x, Y: batch_y, dropout_keep_prob: 1.0})
    avg_cost += loss / total_batch
    predict = np.append(predict, pred)


print ('Test result : ', avg_cost)

plt.plot(testY)
plt.plot(predict)
plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
