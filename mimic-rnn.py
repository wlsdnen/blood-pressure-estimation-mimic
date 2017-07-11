import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Hyper parameters
data_dim = 2
output_dim = 2
max_length = 70
batch_size = 16
num_epochs = 1024

# read and save data
y_data  = np.loadtxt ('./data/dynamic/mimic-sample-target.csv', dtype=float, delimiter=',')
x_data1 = np.loadtxt ('./data/dynamic/mimic-sample-ecg.csv', dtype=float, delimiter=',')
x_data2 = np.loadtxt ('./data/dynamic/mimic-sample-ppg.csv', dtype=float, delimiter=',')
seq_length = np.loadtxt ('./data/dynamic/mimic-sample-length.csv', dtype=int, delimiter=',')

# shuffle data
def ShuffleData (data1, data2, data3, data4):

    shuffle_idx = np.arange(len(data1))
    np.random.shuffle(shuffle_idx)

    data1 = data1[shuffle_idx]
    data2 = data2[shuffle_idx]
    data3 = data3[shuffle_idx]
    data4 = data4[shuffle_idx]

    return data1, data2, data3, data4

y_data, x_data1, x_data2, seq_length = ShuffleData (y_data, x_data1, x_data2, seq_length)

def MergeData (x1, x2):
    merged = []
    for a, b in zip(x1, x2):
        merged.append(np.vstack ((a, b)))
    return np.array(merged).reshape(-1, max_length, data_dim)

x_data = MergeData(x_data1, x_data2)

train_size = int(len(y_data) * 0.7)
test_size = len(y_data) - train_size

trainX, testX = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(y_data)])
trainY, testY = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])
trainSeq, testSeq = np.array(seq_length[0:train_size]), np.array(seq_length[train_size:len(y_data)])

# input place holders
X = tf.placeholder(tf.float32, [None, max_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])
dynamic_length = tf.placeholder(tf.int32, [None])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, forget_bias=1.0, state_is_tuple=True)
#cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, forget_bias=1.0, state_is_tuple=True)

#stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(cell, X, sequence_length=dynamic_length, dtype=tf.float32)

flatten = tf.contrib.layers.flatten(outputs)
print (flatten)

Y_pred = tf.contrib.layers.fully_connected(flatten, output_dim, activation_fn=None)  # We use the last cell's output
#Y_pred = outputs[:, -1]  # We use the last cell's output

# cost/loss
loss = tf.reduce_mean(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.RMSPropOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training batch
for epoch in range(num_epochs+1):

    avg_cost    = 0
    total_batch = int((len(trainX)-1)/batch_size) + 1

    for iteration in range(total_batch):

        start_index = iteration * batch_size
        end_index = min((iteration + 1) * batch_size, train_size)
        batch_x = trainX[start_index:end_index]
        batch_y = trainY[start_index:end_index]
        batch_seq = trainSeq[start_index:end_index]
        
        # A single training step
        _, cost = sess.run([train, loss], feed_dict={X: batch_x, Y: batch_y, dynamic_length: batch_seq})
        avg_cost += cost / total_batch

    if epoch % 500 == 0:
        print('Epoch:', '%03d' % epoch, 'rmse =', avg_cost)

testPredict = sess.run(Y_pred, feed_dict={X: testX, dynamic_length: testSeq})
print (testPredict)

plt.plot(testY)
plt.plot(testPredict)

plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
