import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def MinMaxScaler (_data, _minData, _maxData):
    numerator = _data - _minData
    denominator = _maxData - _minData
    return numerator / (denominator + 1e-7)

def deMinMaxScaler (_data, _minData, _maxData):
    return (_data * (_maxData - _minData + 1e-7)) + _minData

idx, ydata, xdata1, xdata2 = ShuffleData (idx, ydata, xdata1, xdata2)

xdata = np.array([np.concatenate(([x], [y]), axis=0).T for x, y in zip(xdata1, xdata2)] )

ySystolic  = ydata[:, 0]
yDiastolic = ydata[:, 1]

# hyper parameters
data_dim = 2
output_dim = 1

minSystolic = np.min (ySystolic)
maxSystolic = np.max (ySystolic)
ySystolic = MinMaxScaler (ySystolic, minSystolic, maxSystolic)

minDiastolic = np.min (yDiastolic)
maxDiastolic = np.max (yDiastolic)
yDiastolic = MinMaxScaler (yDiastolic, minDiastolic, maxDiastolic)

class Model:

    def __init__(self, sess, name, learning_rate, seq_length):
        self.sess = sess
        self.name = name
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, self.seq_length, data_dim])
            # output place holders
            self.Y = tf.placeholder(tf.float32, [None, 1])

            # Forward passes
            cells = []
            for n in range(5):
                cells.append(tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True))

            stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)
            # cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True)
            outputs, _states = tf.nn.dynamic_rnn(stacked_lstm, self.X, dtype=tf.float32)

            self.Y_pred = outputs[:, -1]  # We use the last cell's output

        # define cost/loss
        self.cost = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # RMSE
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.predictions = tf.placeholder(tf.float32, [None, 1])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_test):
        testPredict = self.sess.run(self.Y_pred, feed_dict={self.X: x_test})
        testPredict = deMinMaxScaler(testPredict, minSystolic, maxSystolic)
        return testPredict

    def get_RMSE(self, y_test, testPred):
        y_test = deMinMaxScaler(y_test, minSystolic, maxSystolic).reshape(-1, 1)
        return self.sess.run(self.rmse, feed_dict={self.targets: y_test, self.predictions: testPred})

    def save_Graph (self, filename, y_test, pred):
        fig = plt.figure()
        tValue = plt.plot(y_test,   label='Real Value')
        pValue = plt.plot(pred,     label = 'Predicted Value')
        plt.xlabel("Time Period")
        plt.ylabel("Blood Pressure")
        plt.savefig(filename)

# initialize
sess = tf.Session()

# hyper parameters
models          = []
num_models      = 1
training_epochs = [500]
alpha           = [0.01]
sequence        = [70]
batch_size      = 100


for m in range(num_models):
    tf.set_random_seed(np.random.randint(1000))
    models.append( Model(sess, "model" + str(m), alpha[m], sequence[m]) )

sess.run(tf.global_variables_initializer())

# train my model
for m_idx, m in enumerate(models):

    train_size  = int(len(ydata) * 0.7)
    test_size   = len(ydata) - train_size

    trainX, testX = np.array(xdata[0:train_size]), np.array(xdata[train_size:len(xdata)])
    trainY, testY = np.array(ySystolic[0:train_size]), np.array(ySystolic[train_size:len(ydata)])

    for epoch in range(training_epochs[m_idx]):

        avg_cost    = 0
        total_batch = int( trainX.shape[0] / batch_size )

        for iter in range(total_batch):

            batch_x = trainX[iter * batch_size : (iter+1) * batch_size]
            batch_y = trainY[iter * batch_size : (iter+1) * batch_size]

            c, _, = m.train(batch_x, batch_y.reshape(-1, 1))
            avg_cost += c / total_batch

        if epoch % 10 == 0:
            print('Epoch:', '%04d' % epoch, 'cost =', avg_cost)

    testPredict     = m.predict(testX)
    rmse_result     = m.get_RMSE(testY, testPredict)
    #
    print(m_idx, 'RMSE: ', rmse_result)
    #
    filename        = '%d-%f-%d-%f' % (m.seq_length, m.learning_rate, training_epochs[m_idx], rmse_result)
    testY_copy      = deMinMaxScaler (testY, minSystolic, maxSystolic).reshape(-1, 1)
    np.savetxt      (filename + '.txt', (testY_copy, testPredict));
    m.save_Graph    (filename + '.png', testY_copy, testPredict)
