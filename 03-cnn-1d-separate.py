import numpy as np
import tensorflow as tf
from mimic_cnn_separate_class import SignalCNN
import data_helper as dh
import matplotlib.pyplot as plt
import pandas as pd

tf.set_random_seed(np.random.randint(10000))  # reproducibility

# Load data
data = np.loadtxt('./data/static/mimic-clean.csv', delimiter=',', dtype=float)
#data = np.load('./data/static/mimic-Part_1.npy')
y_data = data[:, -2:]
x_data1 = data[:, :125]
x_data2 = data[:, 125:-2]

#index = np.load('./data/static/mimic-Part_1-index.npy')
index = np.loadtxt ('./data/static/mimic-clean-index.csv', delimiter=',', dtype=float)

shuffled_idx = dh.ShuffleIndex(index)
y_data = y_data[shuffled_idx]
x_data1 = np.array(x_data1[shuffled_idx]).reshape(-1, 125, 1)
x_data2 = np.array(x_data2[shuffled_idx]).reshape(-1, 125, 1)
index = index[shuffled_idx]

sess = tf.Session()
cnn = SignalCNN(signal_length=125, num_outputs=2, filter_sizes=[3,4,5], num_filters=8)

# Define training procedure
optimizer = tf.train.AdamOptimizer(0.0001)
train_op = optimizer.minimize(cnn.cost)

# Initialize all variables
sess.run(tf.global_variables_initializer())
batch_size = 8192
num_epochs = 1500

# Summary
#summary = tf.summary.merge_all()
#TB_SUMMARY_DIR = './tb/mimic-cnn-1d-separate'
#global_step = 0

## Create summary writer
#writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
#writer.add_graph(sess.graph)

train_size  = int(len(y_data) * 0.7)
test_size   = len(y_data) - train_size

trainX1, testX1 = np.array(x_data1[0:train_size]), np.array(x_data1[train_size:len(x_data1)])
trainX2, testX2 = np.array(x_data2[0:train_size]), np.array(x_data2[train_size:len(x_data2)])
trainY, testY, testIndex = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)]), np.array(index[train_size:len(y_data)])


# Training batch
for epoch in range(num_epochs+1):

    total_cost    = 0
    total_batch = int((len(trainX1)-1)/batch_size) + 1

    batchShuffleIdx = dh.ShuffleIndex(trainY)
    trainX1 = trainX1[batchShuffleIdx]
    trainX2 = trainX2[batchShuffleIdx]
    trainY = trainY[batchShuffleIdx]

    for iteration in range(total_batch):

        start_index = iteration * batch_size
        end_index = min((iteration + 1) * batch_size, train_size)
        batch_x1 = trainX1[start_index:end_index]
        batch_x2 = trainX2[start_index:end_index]
        batch_y = trainY[start_index:end_index]

        # A single training step
        _, loss = sess.run ([train_op, cnn.cost], feed_dict={cnn.input_x1: batch_x1, cnn.input_x2: batch_x2, cnn.input_y: batch_y, cnn.dropout_keep_prob: 1.0})
        total_cost += loss
        #global_step += 1
        #writer.add_summary(s, global_step=global_step)
    
    if epoch % 50 == 0:
        print('Epoch:', '%03d' % epoch, 'cost =', total_cost)


# Test batch
total_cost    = 0
total_batch = int((len(testX1)-1)/batch_size) + 1

predict = np.array([])

for iteration in range(total_batch):

    start_index = iteration * batch_size
    end_index = min((iteration + 1) * batch_size, train_size)
    batch_x1 = testX1[start_index:end_index]
    batch_x2 = testX2[start_index:end_index]
    batch_y = testY[start_index:end_index]

    c, pred = sess.run ([cnn.cost, cnn.predictions], feed_dict={cnn.input_x1: batch_x1, cnn.input_x2: batch_x2, cnn.input_y: batch_y, cnn.dropout_keep_prob: 1.0})
    total_cost += c
    predict = np.append(predict, pred).reshape(-1, 2)

rmse_sbp = dh.GetRMSE(testY[:, 0], predict[:, 0])
rmse_dbp = dh.GetRMSE(testY[:, 1], predict[:, 1])
mean_rmse = np.mean ([rmse_sbp, rmse_dbp])
print ('Test result : ', rmse_sbp, rmse_dbp, mean_rmse)

# Save results
filename        = 'rmse_%2.4f' % (mean_rmse)
result = np.array([testY, predict, testIndex])
np.savetxt(filename + '_y.txt', testY, delimiter=',')
np.savetxt(filename + '_p.txt', predict, delimiter=',')
np.savetxt(filename + '_i.txt', testIndex, delimiter=',')

plt.plot(testY)
plt.plot(predict)
plt.xlabel("Time Period")
plt.ylabel("Blood Pressure")
plt.show()
