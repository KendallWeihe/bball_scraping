import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 1
display_step = 5

# Network Parameters
n_input = 22 # MNIST data input (img shape: 28*28)
# max_size = 400
n_steps = 150 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)
n_hidden_1 = 50 # 1st layer number of features
n_hidden_2 = 50 # 2nd layer number of features

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


def input_data():
    # files = glob.glob('./ncaa_data/*/*.csv')
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    print "Number of files = " + str(len(files))

    input_data = []
    ground_truth = []
    for csv_file in files:
        csv_data = np.genfromtxt(csv_file, delimiter=",")
        # halftime_index = csv_data.shape[0]
        # for i in range(csv_data.shape[0]):
        #     if csv_data[i,0] == 20 or csv_data[i,0] > 20:
        #         halftime_index = i
        #         break
        # # pdb.set_trace()
        # input_data.append(csv_data[0:halftime_index,:])
        #TODO write a data verification script
        if csv_data.shape[0] > 120:
            input_data.append(csv_data[0:n_steps,:])
            diff = csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3]
            if diff > 1:
                ground_truth.append(1)
            elif diff < 1:
                ground_truth.append(0)
            else:
                ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])

    return np.array(input_data), np.array(ground_truth)

# prediction_data = np.genfromtxt("./ncaa_data/Charleston_Villanova.csv", delimiter=",")
# prediction_data = np.genfromtxt("./ncaa_data/Colgate_Penn_State.csv", delimiter=",")
# prediction_data = np.genfromtxt("./ncaa_data/Michigan_South_Carolina.csv", delimiter=",")
# prediction_data = prediction_data[0:n_steps,:].reshape((1,n_steps,n_input))

input_data, ground_truth = input_data()
training_data = input_data[0:45,:]
training_ground_truth = ground_truth[0:45]
prediction_data = input_data[45:51,:]
prediction_ground_truth = ground_truth[45:51]

# Create model
def multilayer_perceptron(x, weights, biases):
    x = tf.reshape(x, [-1, n_steps * n_input])
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
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

# Define loss and optimizer
# pdb.set_trace()
n_samples = tf.cast(tf.shape(x)[0], tf.float32)
# cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
accuracy = tf.abs(tf.sub(pred, y))

# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        acc_batch = []
        loss_batch = []
        # pdb.set_trace()
        for i in range(len(training_data)):
            batch_x = training_data[i].reshape((1, training_data[i].shape[0], training_data[i].shape[1]))
            batch_y = training_ground_truth[i].reshape((1,1))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # if step % display_step == 0:
            # Calculate batch accuracy
            # pdb.set_trace()
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            acc_batch.append(acc[0][0])
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            loss_batch.append(loss)
        print "Step: " + str(step) + "  Accuracy: " + str(np.mean(np.array(acc_batch))) + "  Loss: " + str(np.mean(np.array(loss_batch)))
        step += 1

        pred_vals = []
        # pdb.set_trace()
        for i in range(len(prediction_data)):
            batch_x = prediction_data[i].reshape((1, prediction_data[i].shape[0], prediction_data[i].shape[1]))
            # batch_y = prediction_ground_truth[i]
            pred_val = sess.run(tf.nn.softmax(pred), feed_dict={x: batch_x})
            pred_vals.append(pred_val[0][0])
            print "Prediction = " + str(pred_val) + "  Actual = " + str(prediction_ground_truth[i])


        if np.mean(np.array(acc_batch)) < 11:
            prediction_ground_truth = prediction_ground_truth.tolist()
            plot_data = []
            for i in range(len(pred_vals)):
                plot_data.append([prediction_ground_truth[i], pred_vals[i]])
            # plot_data = [prediction_ground_truth, pred_vals]
            # plt.plot(plot_data, 'ro')
            plt.scatter(prediction_ground_truth, pred_vals)
            plt.grid(True)
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')
            plt.show()
            prediction_ground_truth = np.array(prediction_ground_truth)

    print("Optimization Finished!")
