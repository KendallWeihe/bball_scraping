import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt
from scipy.stats import linregress

learning_rate = 0.001
training_iters = 100000
batch_size = 20
display_step = 5

n_input = 22
n_steps = 115
n_hidden = 64
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


def input_data():
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    print "Number of files = " + str(len(files))

    input_data = []
    ground_truth = []
    scores = []
    for csv_file in files:
        csv_data = np.genfromtxt(csv_file, delimiter=",")
        if csv_data.shape[0] > n_steps:
            input_data.append(csv_data[0:n_steps,:])
            if csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3] < 0:
                ground_truth.append(0)
            else:
                ground_truth.append(1)
            scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])

    return np.array(input_data), np.array(ground_truth), np.array(scores)

input_data, ground_truth, scores = input_data()
training_data = input_data[0:input_data.shape[0] - n_predictions - n_regression_points, :]
training_ground_truth = ground_truth[0:ground_truth.shape[0] - n_predictions - n_regression_points]
prediction_data = input_data[input_data.shape[0] - n_predictions - n_regression_points:input_data.shape[0] - n_regression_points, :]
prediction_ground_truth = ground_truth[ground_truth.shape[0] - n_predictions - n_regression_points:ground_truth.shape[0] - n_regression_points]
regression_data = input_data[input_data.shape[0] - n_regression_points: input_data.shape[0], :]
regression_ground_truth = ground_truth[ground_truth.shape[0] - n_regression_points: ground_truth.shape[0]]

def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.dropout(output, 0.75)

pred = RNN(x, weights, biases)
cost = tf.nn.softmax_cross_entropy_with_logits(pred, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.abs(tf.sub(tf.nn.softmax(pred), y))
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    for j in range(10000):

        start_pos = np.random.randint(len(training_data) - batch_size)
        batch_x = training_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = training_ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        # print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)

        # if step % 50 == 0:
        pred_vals = []
        for i in range(prediction_data.shape[0]):
            pred_val = sess.run(tf.nn.softmax(pred), feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
            # print "Prediction = " + str(np.mean(pred_val)) + "  Actual = " + str(prediction_ground_truth[i])
            print str(np.mean(pred_val)) + "," + str(prediction_ground_truth[i])

        print "\n"
        step += 1

    print("Optimization Finished!")
