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
dropout = 0.75

n_input = 22
n_steps = 115
n_classes = 1
n_predictions = 30
n_regression_points = 0

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

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def CNN(x, weights, biases):
    x = tf.reshape(x, [-1,n_steps,n_input,1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.softmax(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([7, 7, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([7, 7, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([25*1*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = CNN(x, weights, biases)
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
            #     save_path = "./lstm_models/lstm_model_n_steps_" + str(n_steps) + "_" + str(step) + "_" + str(r_value**2) + ".ckpt"
            #     saver.save(sess, save_path)
            #     min_diff = np.mean(np.abs(np.array(pred_val - prediction_ground_truth)))
    # plt.plot(accuracy_data)
    # plt.show()
    print("Optimization Finished!")
