import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt
from scipy.stats import linregress

learning_rate = 0.001
training_iters = 100000
batch_size = 10
display_step = 5

n_input = 22
n_steps = 175
n_hidden = 500
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

single_game_data = np.genfromtxt("./ncaa_data/Hofstra_Kentucky.csv", delimiter=",")
prediction_data = single_game_data[0:n_steps,:].reshape((1,n_steps,n_input))

def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.dropout(output, 0.75)

pred = RNN(x, weights, biases)
n_samples = tf.cast(tf.shape(x)[0], tf.float32)
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.abs(tf.sub(pred, y))
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./lstm_models/200/lstm_model_n_steps_175_4500_0.807734028939_1.3577159583_-1.4186952694.ckpt")
    pred_val = sess.run(pred, feed_dict={x: prediction_data})
    print "Prediction = " + str(pred_val)
