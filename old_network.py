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
batch_size = 4
display_step = 5

# Network Parameters
n_input = 22 # MNIST data input (img shape: 28*28)
# max_size = 400
n_steps = 140 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)

prediction_data = np.genfromtxt("./ncaa_data/Iowa_State_Miami.csv", delimiter=",")
n_steps = prediction_data.shape[0]
prediction_data = prediction_data[0:n_steps,:].reshape((1,n_steps,n_input))

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def input_data():
    # files = glob.glob('./ncaa_data/*/*.csv')
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    print "Number of files = " + str(len(files))

    input_data = []
    ground_truth = []
    scores = []
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
        if csv_data.shape[0] > n_steps:
            input_data.append(csv_data[0:n_steps,:])
            ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])
            scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])

    return np.array(input_data), np.array(ground_truth), np.array(scores)

# prediction_data = np.genfromtxt("./ncaa_data/Iowa_State_Miami.csv", delimiter=",")
# prediction_data = np.genfromtxt("./ncaa_data/Colgate_Penn_State.csv", delimiter=",")


input_data, ground_truth, scores = input_data()
training_data = input_data[0:58,:]
training_ground_truth = ground_truth[0:58]
# prediction_data = input_data[58:66,:]
# prediction_ground_truth = ground_truth[58:66]
# prediction_scores = scores[58:66]

def RNN(x, weights, biases):

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.nn.dropout(output, 0.75)

pred = RNN(x, weights, biases)

# Define loss and optimizer
# pdb.set_trace()
n_samples = tf.cast(tf.shape(x)[0], tf.float32)
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
accuracy = tf.abs(tf.sub(pred, y))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    min_diff = 17
    # Keep training until reach max iterations
    avg_diff = []
    avg_pred = []
    # while step * batch_size < training_iters:
    for j in range(1250):
        acc_batch = []
        loss_batch = []
        # pdb.set_trace()
        start_pos = np.random.randint(len(training_data) - batch_size)
        # for i in range(start_pos,start_pos+4):
        # for i in range(len(input_data)):
        # pdb.set_trace()
        batch_x = training_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
            # batch_x = input_data[i].reshape((1, input_data[i].shape[0], input_data[i].shape[1]))
        batch_y = training_ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
            # batch_y = ground_truth[i].reshape((1,1))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # if step % display_step == 0:
            # Calculate batch accuracy
            # pdb.set_trace()
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        acc_batch.append(acc[0][0])
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        loss_batch.append(loss)
        # print "Step: " + str(step) + "  Accuracy: " + str(np.mean(np.array(acc_batch))) + "  Loss: " + str(np.mean(np.array(loss_batch)))
        print "Step: " + str(step) + "  Accuracy: " + str(acc[0][0]) + "  Loss: " + str(loss)
        step += 1

        pred_val = sess.run(pred, feed_dict={x: prediction_data.reshape((1,n_steps,n_input))})
        avg_pred.append(pred_val[0][0])
        print "Prediction = " + str(pred_val[0][0])
        print "Prediction average = " + str(np.mean(np.array(avg_pred)))

        # if step % 50 == 0:
        #     pred_vals = []
        #     local_avg = []
        #     # pdb.set_trace()
        #     for i in range(len(prediction_data)):
        #         batch_x = prediction_data[i].reshape((1, prediction_data[i].shape[0], prediction_data[i].shape[1]))
        #         batch_y = prediction_ground_truth[i]
        #         pred_val = sess.run(pred, feed_dict={x: batch_x})
        #         avg_diff.append(pred_val[0][0] - prediction_ground_truth[i])
        #         local_avg.append(pred_val[0][0] - prediction_ground_truth[i])
        #         pred_vals.append(pred_val[0][0])
        #         print "Prediction = " + str(pred_val) + "  Actual = " + str(prediction_ground_truth[i]) + "  Score difference = " + str(prediction_scores[i])
        # 
        #     # pred_val = sess.run(pred, feed_dict={x: prediction_data})
        #     # print "Prediction = " + str(pred_val)
        #     # avg_diff.append(pred_val)
        #     print "Average = " + str(np.mean(np.abs(np.array(avg_diff))))
        #     #print "Average difference = " + str(np.mean(np.abs(np.array(pred_val - prediction_ground_truth))))
        #     print "Local average = " + str(np.mean(np.abs(np.array(local_avg))))

        # if step % 50 == 0:
    # prediction_ground_truth = prediction_ground_truth.tolist()
    # plot_data = []
    # for i in range(len(pred_vals)):
    #     plot_data.append([prediction_ground_truth[i], pred_vals[i]])
    # # plot_data = [prediction_ground_truth, pred_vals]
    # # plt.plot(plot_data, 'ro')
    # plt.scatter(prediction_ground_truth, pred_vals)
    # plt.grid(True)
    # plt.axhline(0, color='black')
    # plt.axvline(0, color='black')
    # plt.axis((-40,40,-40,40))
    # plt.show()
    # # pdb.set_trace()
    # prediction_ground_truth = np.array(prediction_ground_truth)

        #if np.mean(np.abs(np.array(pred_val - prediction_ground_truth))) < min_diff:
        #    save_path = "./lstm_models/lstm_model_" + str(step) + ".ckpt"
        #    saver.save(sess, save_path)
        #    min_diff = np.mean(np.abs(np.array(pred_val - prediction_ground_truth)))

    # save_path = "./lstm_models/lstm_model_140.ckpt"
    # saver.save(sess, save_path)

    print("Optimization Finished!")
