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


def input_data():
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    print "Number of files = " + str(len(files))

    input_data = []
    ground_truth = []
    scores = []
    for csv_file in files:
        try:
            csv_data = np.genfromtxt(csv_file, delimiter=",")
            if csv_data.shape[0] > n_steps:
                input_data.append(csv_data[0:n_steps,:])
                ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])
                scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])
        except:
            print csv_file

    return np.array(input_data), np.array(ground_truth), np.array(scores)

input_data, ground_truth, scores = input_data()
randomize = np.arange(len(input_data))
np.random.shuffle(randomize)
input_data = input_data[randomize]
ground_truth = ground_truth[randomize]

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
    avg_pred_vals = []
    avg_reg_vals = []
    max_r_values = [0,0,0]
    accuracy_data = []
    for step in range(10000):

        start_pos = np.random.randint(len(training_data) - batch_size)
        batch_x = training_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = training_ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        accuracy_data.append(np.mean(acc))
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        # print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)

        # if step % 50 == 0:
        pred_vals = []
        for i in range(prediction_data.shape[0]):
            pred_val = sess.run(pred, feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
            # pdb.set_trace()
            pred_vals.append(int(pred_val))
            avg_pred_diff.append(int(pred_val)-prediction_ground_truth[i])
            # print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(prediction_ground_truth[i])
        # print "Prediction average = " + str(np.mean(np.abs(np.array(avg_pred_diff))))

        slope, intercept, r_value, p_value, std_err = linregress(np.array(pred_vals), prediction_ground_truth)
        # print "Slope = " + str(slope)
        # print "R^2 = " + str(r_value**2)

        # if step % 25 == 0:
        print "Step: " + str(step) + "  Mean Accuracy: " + str(np.mean(accuracy_data)) + "  Loss: " + str(loss)

        # if step > 750 and r_value**2 > 0.63 and r_value**2 > np.min(np.array(max_r_values)):
        # if step > 750 and r_value**2 > 0.5:
        if step > 500:
            print "Step = " + str(step)
            print "Accuracy = " + str(np.mean(acc))

            avg_pred_vals.append(pred_vals)
            avg_pred_vals_np = np.array(avg_pred_vals)
            avg_vals = []
            print "Raw prediction data ----------------------------------------------"
            for i in range(avg_pred_vals_np.shape[1]):
                avg_vals.append(np.mean(avg_pred_vals_np[:,i]))
                print str(np.mean(avg_pred_vals_np[:,i])) + "," + str(prediction_ground_truth[i])
                # print "Average for game " + str(i) + " = " + str(np.mean(avg_pred_vals_np[:,i])) + "  Actual = " + str(prediction_ground_truth[i])
            print "Slope before average = " + str(slope)
            print "R^2 before average = " + str(r_value**2)
            slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), prediction_ground_truth)
            print "Slope after average = " + str(slope)
            print "R^2 after average = " + str(r_value**2)
            print "Intercept after average = " + str(intercept)
            max_r_values.append(r_value**2)
            if len(max_r_values) > 8:
                del max_r_values[np.where(np.array(max_r_values) == np.min(np.array(max_r_values)))[0][0]]

            print "Actual prediction values --------------------------------------------"
            for i in range(len(avg_vals)):
                actual_value = slope * avg_vals[i] + intercept
                print str(actual_value) + "," + str(prediction_ground_truth[i])

                # linear_reg_vals = np.array([slope, intercept])
                # filename = "./lstm_models/linear_reg_vals_" + str(step) + "_" + str(r_value**2) + ".csv"
                # np.savetxt(filename, linear_reg_vals, delimiter=",")

                # pred_vals = []
                # for i in range(regression_data.shape[0]):
                #     pred_val = sess.run(pred, feed_dict={x: regression_data[i].reshape((1, n_steps, n_input))})
                #     pred_vals.append(pred_val[0][0])
                # avg_reg_vals.append(pred_vals)
                # avg_vals = []
                # print "Regression data --------------------------------------------------------"
                # for i in range(len(pred_vals)):
                #     avg_vals.append(np.mean(np.array(avg_reg_vals)[:,i]))
                #     # print "Average for game " + str(i) + " = " + str(avg_vals[i]) + " Actual = " + str(regression_ground_truth[i])
                #     print str(avg_vals[i]) + "," + str(regression_ground_truth[i])
                # slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), regression_ground_truth)
                # print "Slope after average = " + str(slope)
                # print "R^2 after average = " + str(r_value**2)
                # print "Intercept after average = " + str(intercept)
                # print "Actual regression values -------------------------------------------------- "
                # for i in range(len(avg_vals)):
                #     actual_value = slope * avg_vals[i] + intercept
                #     print str(actual_value) + "," + str(regression_ground_truth[i])

            print "\n"

            step += 1

            # if step % 250 == 0:
            #     save_path = "./lstm_models/200/lstm_model_n_steps_" + str(n_steps) + "_" + str(step) + "_" + str(r_value**2) + ".ckpt"
            #     saver.save(sess, save_path)
            #     min_diff = np.mean(np.abs(np.array(pred_val - prediction_ground_truth)))
    # plt.plot(accuracy_data)
    # plt.show()
    print("Optimization Finished!")
