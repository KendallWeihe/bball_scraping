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

n_predictions = 30
n_regression_points = 0

# Network Parameters
n_input = 22 # MNIST data input (img shape: 28*28)
# max_size = 400
n_steps = 115 # timesteps
n_classes = 1 # MNIST total classes (0-9 digits)
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

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
training_data = input_data[0:input_data.shape[0] - n_predictions - n_regression_points, :]
training_ground_truth = ground_truth[0:ground_truth.shape[0] - n_predictions - n_regression_points]
prediction_data = input_data[input_data.shape[0] - n_predictions - n_regression_points:input_data.shape[0] - n_regression_points, :]
prediction_ground_truth = ground_truth[ground_truth.shape[0] - n_predictions - n_regression_points:ground_truth.shape[0] - n_regression_points]
regression_data = input_data[input_data.shape[0] - n_regression_points: input_data.shape[0], :]
regression_ground_truth = ground_truth[ground_truth.shape[0] - n_regression_points: ground_truth.shape[0]]

# Create model
def multilayer_perceptron(x, weights, biases):
    x = tf.reshape(x, [-1, n_steps * n_input])
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.softmax(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.softmax(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    dropout_layer = tf.nn.dropout(out_layer, 0.75)
    return dropout_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_steps * n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)
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
            pred_val = sess.run(pred, feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
            pred_vals.append(pred_val[0][0])
            avg_pred_diff.append(pred_val[0][0]-prediction_ground_truth[i])
            # print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(prediction_ground_truth[i])
        # print "Prediction average = " + str(np.mean(np.abs(np.array(avg_pred_diff))))

        slope, intercept, r_value, p_value, std_err = linregress(np.array(pred_vals), prediction_ground_truth)
        # print "Slope = " + str(slope)
        # print "R^2 = " + str(r_value**2)

        if step % 50 == 0 and step < 750:
            print "Step = " + str(step)

        if step > 750:
            print "Step = " + str(step)
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

            print "Actual prediction values --------------------------------------------"
            for i in range(len(avg_vals)):
                actual_value = slope * avg_vals[i] + intercept
                print str(actual_value) + "," + str(prediction_ground_truth[i])

            pred_vals = []
            for i in range(regression_data.shape[0]):
                pred_val = sess.run(pred, feed_dict={x: regression_data[i].reshape((1, n_steps, n_input))})
                pred_vals.append(pred_val[0][0])
            avg_reg_vals.append(pred_vals)
            avg_vals = []
            print "Regression data --------------------------------------------------------"
            for i in range(len(pred_vals)):
                avg_vals.append(np.mean(np.array(avg_reg_vals)[:,i]))
                # print "Average for game " + str(i) + " = " + str(avg_vals[i]) + " Actual = " + str(regression_ground_truth[i])
                print str(avg_vals[i]) + "," + str(regression_ground_truth[i])
            slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), regression_ground_truth)
            print "Slope after average = " + str(slope)
            print "R^2 after average = " + str(r_value**2)
            print "Intercept after average = " + str(intercept)
            print "Actual values: "
            for i in range(len(avg_vals)):
                actual_value = slope * avg_vals[i] + intercept
                print str(actual_value) + "," + str(regression_ground_truth[i])

            print "\n"

        step += 1

    print("Optimization Finished!")
