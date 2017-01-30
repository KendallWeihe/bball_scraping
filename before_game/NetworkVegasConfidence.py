import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.ops import rnn, rnn_cell
import os
from scipy.stats import linregress
import math
import sys
import csv

# ARGV:
#     prediction file name and path -- located in stats/w_out_score/
    # date
if len(sys.argv) != 3:
    print "Missing arguments"

# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 10
n_predictions = 50
batch_size = 50


# Training Data
path = "./withNetworkPrediction/"
files = os.listdir(path)
files.sort()
data = np.genfromtxt(path+files[0], delimiter=",")
for i in range(len(files)):
    temp = np.genfromtxt(path+files[i], delimiter=",")
    data = np.concatenate((data, temp), axis=0)

groundTruth = data[:,30].copy()
data = data[:,0:30].copy()

n_input = data.shape[1]
n_classes = 1
dropout = 0.75

randomize = np.arange(len(data))
np.random.shuffle(randomize)
data = data[randomize]

train_X = data[0:data.shape[0]-n_predictions,:]
train_Y = groundTruth[0:groundTruth.shape[0]-n_predictions]
pred_X = data[data.shape[0]-n_predictions:data.shape[0],:]
pred_Y = groundTruth[groundTruth.shape[0]-n_predictions:groundTruth.shape[0]]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
n_hidden = 128
n_steps = 1
def network(x):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.dropout(output, dropout)

# Store layers weight & bias
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = network(x)
predNode = tf.nn.softmax(pred)

# Construct model
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Fit all training data
    for epoch in range(training_epochs):
        start_pos = np.random.randint(len(train_X) - batch_size)
        batch_x = train_X[start_pos:start_pos+batch_size].reshape((batch_size, n_input))
        batch_y = train_Y[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        pred_vals = []
        correctCount = 0.0
        for i in range(len(pred_X)):
            pred_val = sess.run(predNode, feed_dict={x: pred_X[i].reshape((1,n_input)), keep_prob:1.0})
            pred_vals.append(float(pred_val))
            if float(pred_val) == pred_Y[i]:
                correctCount = correctCount + 1.0

        print "Epoch = " + str(epoch)
        print "Loss = " + str(np.mean(np.absolute(loss)))
        print "Accuracy = " + str(correctCount/float(n_predictions))
        print "\n"
