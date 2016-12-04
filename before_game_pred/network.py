'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.ops import rnn, rnn_cell

# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 10
n_predictions = 5
batch_size = 3

# Training Data
data = np.genfromtxt("./stats/w_score/2-161203.csv", delimiter=",")

# for i in range(data.shape[0]):
#     if data[i,36] > data[i,37]:
#         data[i,36] = 1
#         data[i,37] = 0
#     else:
#         data[i,36] = 0
#         data[i,37] = 0

train_X = data[0:data.shape[0]-n_predictions,0:36]
train_Y = data[0:data.shape[0]-n_predictions,36] - data[0:data.shape[0]-n_predictions,37]

pred_X = data[data.shape[0]-n_predictions:data.shape[0],0:36]
pred_Y = data[data.shape[0]-n_predictions:data.shape[0],36] - data[data.shape[0]-n_predictions:data.shape[0],37]

# Network Parameters
# n_hidden_1 = 65 # 1st layer number of features
# n_hidden_2 = 65 # 2nd layer number of features
# n_input = 36 # MNIST data input (img shape: 28*28)
# n_classes = 1 # MNIST total classes (0-9 digits)
#
# # tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])


# Create model
# def multilayer_perceptron(x, weights, biases):
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#     return tf.nn.dropout(out_layer, 0.75)
#
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

n_input = 36
n_steps = 1
n_hidden = 175
n_classes = 1
n_predictions = 30
n_regression_points = 0

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.dropout(output, 0.75)

pred = RNN(x, weights, biases)

# Construct model
n_samples = tf.cast(tf.shape(x)[0], tf.float32)
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.abs(tf.sub(pred, y))
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        start_pos = np.random.randint(len(train_X) - batch_size)
        batch_x = train_X[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = train_Y[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

        print "Loss = " + str(np.mean(np.absolute(loss)))
        print "Accuracy = " + str(np.mean(np.absolute(acc)))

        for i in range(len(pred_X)):
            pred_val = sess.run(pred, feed_dict={x: pred_X[i].reshape((1,1,36))})
            print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(pred_Y[i])
