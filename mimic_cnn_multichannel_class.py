import tensorflow as tf

class SignalCNN(object):
    # A CNN for signal regression.

    def __init__(self, signal_length, num_outputs, filter_sizes, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, signal_length, 2], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_outputs], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        pooled_outputs = []

        for i, filter_size in enumerate (filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolutional Layer
                conv1 = tf.layers.conv1d (
                inputs=self.input_x,
                filters=num_filters,
                kernel_size=[filter_size],
                padding='VALID',
                activation=tf.nn.relu)
                print ('conv1', conv1)

                # pool_size = tf.int32(filter_size/2)
                # Max-pooling over the outputs
                pooled1 = tf.layers.max_pooling1d(
                inputs=conv1,
                pool_size=2,
                strides=1,
                padding='VALID')
                pooled1 = tf.nn.dropout(pooled1, self.dropout_keep_prob)
                print ('pool1', pooled1)

                # Convolutional Layer
                conv2 = tf.layers.conv1d (
                inputs=pooled1,
                filters=num_filters,
                kernel_size=[filter_size],
                padding='VALID',
                activation=tf.nn.relu)
                print ('conv2', conv2)

                # Max-pooling over the outputs
                pooled2 = tf.layers.max_pooling1d(
                inputs=conv2,
                pool_size=2,
                strides=1,
                padding='VALID')
                pooled2 = tf.nn.dropout(pooled2, self.dropout_keep_prob)
                print ('pool2', pooled2)

                pooled2 = tf.contrib.layers.flatten(pooled2)
                pooled_outputs.append(pooled2)

        # Combine all the pooled features
        self.h_pool_flat = tf.concat(pooled_outputs, 1)
        print ('h_pool_flat', self.h_pool_flat)
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        
        with tf.name_scope("fully_connected_layer1"):
            self.fclayer1 = tf.contrib.layers.fully_connected(self.h_drop, 8192, activation_fn=tf.nn.relu)
            self.h_drop1 = tf.nn.dropout(self.fclayer1, self.dropout_keep_prob)

        with tf.name_scope("fully_connected_layer2"):
            self.fclayer2 = tf.contrib.layers.fully_connected(self.h_drop1, 4096, activation_fn=tf.nn.relu)
            self.h_drop2 = tf.nn.dropout(self.fclayer2, self.dropout_keep_prob)

        with tf.name_scope("fully_connected_layer3"):
            self.fclayer3 = tf.contrib.layers.fully_connected(self.h_drop2, 2048, activation_fn=tf.nn.relu)
            self.h_drop3 = tf.nn.dropout(self.fclayer3, self.dropout_keep_prob)

        with tf.name_scope("fully_connected_layer4"):
            self.fclayer4 = tf.contrib.layers.fully_connected(self.h_drop3, 1024, activation_fn=tf.nn.relu)
            self.h_drop4 = tf.nn.dropout(self.fclayer4, self.dropout_keep_prob)

        #with tf.name_scope("fully_connected_layer5"):
        #    self.fclayer5 = tf.contrib.layers.fully_connected(self.h_drop4, 512, activation_fn=tf.nn.relu)
        #    self.h_drop5 = tf.nn.dropout(self.fclayer5, self.dropout_keep_prob)
        #
        # with tf.name_scope("fully_connected_layer6"):
        #     self.fclayer6 = tf.contrib.layers.fully_connected(self.h_drop5, 2500, activation_fn=tf.nn.relu)
        #     self.h_drop6 = tf.nn.dropout(self.fclayer6, self.dropout_keep_prob)
        #
        # with tf.name_scope("fully_connected_layer7"):
        #     self.fclayer7 = tf.contrib.layers.fully_connected(self.h_drop6, 1250, activation_fn=tf.nn.relu)
        #     self.h_drop7 = tf.nn.dropout(self.fclayer7, self.dropout_keep_prob)
        #
        # with tf.name_scope("fully_connected_layer8"):
        #     self.fclayer8 = tf.contrib.layers.fully_connected(self.h_drop7, 625, activation_fn=tf.nn.relu)
        #     self.h_drop8 = tf.nn.dropout(self.fclayer8, self.dropout_keep_prob)

        with tf.name_scope("output_layer"):
            self.predictions = tf.contrib.layers.fully_connected(self.h_drop4, num_outputs, activation_fn=None)

        with tf.name_scope("loss"):
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.predictions - self.input_y)))
