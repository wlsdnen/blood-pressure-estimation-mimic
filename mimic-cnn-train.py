import numpy as np
import tensorflow as tf
from mimic_cnn_class import SignalCNN
import data_helper as dh
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

# Load data
data = np.loadtxt ('./data/mimic-all.csv', delimiter=',', dtype=float)
x_data = []
y_data = data[:, [-1]]
x_data1 = data[:, :70]
x_data2 = data[:, 70:-2]

y_data, x_data1, x_data2 = dh.ShuffleData(_ydata=y_data, _xdata1=x_data1, _xdata2=x_data2)
x_data = dh.MergeData(x1=x_data1, x2=x_data2)
print ('x_data shape :', x_data.shape)

sess = tf.Session()
cnn = SignalCNN(signal_length=70, num_outputs=1, filter_sizes=[2, 4], num_filters=64)

# Define training procedure
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(cnn.cost)

# Initialize all variables
sess.run(tf.global_variables_initializer())

batch_size = 8192
num_epochs = 512

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
        _, loss = sess.run ([train_op, cnn.cost], feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: 0.5})
        avg_cost += loss / total_batch

    if epoch % 5 == 0:
        print('Epoch:', '%03d' % epoch, 'rmse =', avg_cost)

avg_cost    = 0
total_batch = int((len(testX)-1)/batch_size) + 1

predict = np.array([])

for iteration in range(total_batch):

    start_index = iteration * batch_size
    end_index = min((iteration + 1) * batch_size, train_size)
    batch_x = testX[start_index:end_index]
    batch_y = testY[start_index:end_index]

    a, pred = sess.run ([cnn.cost, cnn.predictions], feed_dict={cnn.input_x: batch_x, cnn.input_y: batch_y, cnn.dropout_keep_prob: 1.0})
    avg_cost += loss / total_batch
    predict = np.append(predict, pred)


print ('Test result : ', avg_cost)

plt.plot(testY)
plt.plot(predict)
plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
