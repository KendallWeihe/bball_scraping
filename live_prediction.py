import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt

learning_rate = 0.001
training_iters = 100000
batch_size = 4
display_step = 5

n_input = 22
n_hidden = 128
n_classes = 1

prediction_data = np.genfromtxt("./ncaa_data/Iowa_State_Miami.csv", delimiter=",")
n_steps = prediction_data.shape[0]
prediction_data = prediction_data[0:n_steps,:].reshape((1,n_steps,n_input))

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
            ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])
            scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])

    return np.array(input_data), np.array(ground_truth), np.array(scores)

input_data, ground_truth, scores = input_data()

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
    step = 1
    avg_pred = []
    for j in range(1250):

        start_pos = np.random.randint(len(input_data) - batch_size)
        batch_x = input_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)

        pred_val = sess.run(pred, feed_dict={x: prediction_data.reshape((1,n_steps,n_input))})
        avg_pred.append(pred_val[0][0])
        print "Prediction = " + str(pred_val[0][0])
        print "Prediction average = " + str(np.mean(np.array(avg_pred)))
        print "\n"

        step += 1

    print("Optimization Finished!")
