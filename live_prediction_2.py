import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt
from scipy.stats import linregress

learning_rate = 0.001
training_iters = 100000
batch_size = 3
display_step = 5

n_input = 22
n_hidden = 64
n_steps = 121
n_classes = 1
n_predictions = 30

single_game_data = np.genfromtxt("./ncaa_data/Rutgers_Miami.csv", delimiter=",")
n_steps = single_game_data.shape[0]
single_game_data = single_game_data[0:n_steps,:].reshape((1,n_steps,n_input))

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
        try:
            csv_data = np.genfromtxt(csv_file, delimiter=",")
            if csv_data.shape[0] >= n_steps:
                input_data.append(csv_data[0:n_steps,:])
                ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])
                scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])
        except:
            print csv_file

    return np.array(input_data), np.array(ground_truth), np.array(scores)

input_data, ground_truth, scores = input_data()
training_data = input_data[0:input_data.shape[0] - n_predictions,:]
training_ground_truth = ground_truth[0:ground_truth.shape[0] - n_predictions]
prediction_data = input_data[input_data.shape[0] - n_predictions:input_data.shape[0]]
prediction_ground_truth = ground_truth[ground_truth.shape[0] - n_predictions:ground_truth.shape[0]]

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
    avg_pred_diff = []
    r_values = []
    avg_pred_vals = []
    for j in range(7500):
        start_pos = np.random.randint(len(training_data) - batch_size)
        batch_x = training_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = training_ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        # print "\n"
        # print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)

        pred_vals = []
        for i in range(prediction_data.shape[0]):
            pred_val = sess.run(pred, feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
            pred_vals.append(pred_val[0][0])
            avg_pred_diff.append(pred_val[0][0]-prediction_ground_truth[i])
            # print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(prediction_ground_truth[i])
        # print "Prediction average = " + str(np.mean(np.abs(np.array(avg_pred_diff))))
        slope, intercept, r_value, p_value, std_err = linregress(np.array(pred_vals), prediction_ground_truth)
        # print "Slope = " + str(slope)
        # print "R^2 = " + str(r_value**2)

        if step % 200 == 0:
            print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)

        # if r_value**2 > 0.48:
        if step > 50:
            print "Step = " + str(step)
            print "Num steps = " + str(n_steps)
            print "Accuracy = " + str(acc[0][0])

            print "Slope before average = " + str(slope)
            print "R^2 before average = " + str(r_value**2)

            avg_pred_vals.append(pred_vals)
            avg_pred_vals_np = np.array(avg_pred_vals)
            avg_vals = []
            print "Raw values ----------------------------------"
            for i in range(avg_pred_vals_np.shape[1]):
                avg_vals.append(np.mean(avg_pred_vals_np[:,i]))
                print str(np.mean(avg_pred_vals_np[:,i])) + "," + str(prediction_ground_truth[i])
                # print "Average for game " + str(i) + " = " + str(np.mean(avg_pred_vals_np[:,i])) + "  Actual = " + str(prediction_ground_truth[i])
            slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), prediction_ground_truth)
            print "Slope after average = " + str(slope)
            print "R^2 after average = " + str(r_value**2)
            print "Intercept after average = " + str(intercept)

            print "Actual values -------------------------------"
            for i in range(len(avg_vals)):
                actual_value = slope * avg_vals[i] + intercept
                print str(actual_value) + "," + str(prediction_ground_truth[i])

            pred_val = sess.run(pred, feed_dict={x: single_game_data})
            print "Single game prediction = " + str(pred_val[0][0])
            r_values.append(pred_val[0][0])
            print "Average single game prediction = " + str(np.mean(np.array(r_values)))

            actual_value = slope * np.mean(np.array(r_values)) + intercept
            print "Actual spread = " + str(actual_value)

            print "\n"
        step += 1

    print("Optimization Finished!")
