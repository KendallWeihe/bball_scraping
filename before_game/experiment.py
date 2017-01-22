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
#     experiment file name and path -- located in stats/w_score/

if len(sys.argv) != 2:
    print "Missing arguments"

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

spreads = np.genfromtxt("./testing.csv", delimiter=",")

# Parameters
learning_rate = 0.01
dropout = 0.75
training_epochs = 300
display_step = 10
n_predictions = 50
batch_size = 50

# Training Data
path = "stats/w_score/"
files = os.listdir(path)
files.sort()
data = np.genfromtxt(path+files[0], delimiter=",")
for i in range(len(files)):
    if "stats/w_out_score/"+files[i] != sys.argv[1]:
        temp = np.genfromtxt(path+files[i], delimiter=",")
        data = np.concatenate((data, temp), axis=0)

for i in range(data.shape[1]-2):
    data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))

predictionFile = np.genfromtxt(sys.argv[1], delimiter=",")
scores = predictionFile[:,36] - predictionFile[:,37]
predictionFile = predictionFile[:,0:36]

for i in range(data.shape[1]-2):
    predictionFile[:,i] = (predictionFile[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))
    data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))

overallAvgPreds = []
avgPredDiff = []
for k in range(500):
    print "Session number: " + str(k)

    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]

    train_X = data[0:data.shape[0]-n_predictions,0:36]
    train_Y = data[0:data.shape[0]-n_predictions,36] - data[0:data.shape[0]-n_predictions,37]

    pred_X = data[data.shape[0]-n_predictions:data.shape[0],0:36]
    pred_Y = data[data.shape[0]-n_predictions:data.shape[0],36] - data[data.shape[0]-n_predictions:data.shape[0],37]

    n_input = 36
    n_classes = 1

    # Network Parameters
    n_hidden_1 = 50 # 1st layer number of features
    n_hidden_2 = 50 # 2nd layer number of features
    n_hidden_3 = 50
    n_hidden_4 = 50

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Create model
    n_hidden = 128
    n_steps = 1
    def multilayer_perceptron(x, weights, biases):
        # x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(0, n_steps, x)
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
        all_lstm_outputs = tf.reshape(tf.stack(outputs, axis=1), [-1, n_steps*n_hidden])
        output = tf.matmul(outputs[-1], weights['out']) + biases['out']
        # output = tf.matmul(all_lstm_outputs, weights['all_out']) + biases['out']
        return tf.nn.dropout(output, 0.75)
        # Hidden layer with RELU activation
        # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # # layer_1 = tf.nn.softmax(layer_1)
        # layer_1 = tf.nn.relu(layer_1)
        # layer_1 = tf.nn.l2_normalize(layer_1, 1)
        # layer_1 = tf.nn.dropout(layer_1, dropout)

        # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # # layer_2 = tf.nn.softmax(layer_2)
        # layer_2 = tf.nn.relu(layer_2)
        # layer_2 = tf.nn.l2_normalize(layer_2, 1)
        # # layer_2 = tf.nn.dropout(layer_2, dropout)
        #
        # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        # # layer_3 = tf.nn.softmax(layer_3)
        # layer_3 = tf.nn.relu(layer_3)
        # layer_3 = tf.nn.l2_normalize(layer_3, 1)
        # # layer_3 = tf.nn.dropout(layer_3, dropout)
        #
        # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        # # layer_4 = tf.nn.softmax(layer_4)
        # layer_4 = tf.nn.relu(layer_4)
        # layer_4 = tf.nn.l2_normalize(layer_4, 1)
        # layer_4 = tf.nn.dropout(layer_4, dropout)

        # out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
        # return tf.nn.dropout(out_layer, dropout)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Construct model
    n_samples = tf.cast(tf.shape(x)[0], tf.float32)
    cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    accuracy = tf.reduce_mean(tf.abs(tf.sub(pred, y)))
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        avg_pred_vals = []
        avgPredVals = []
        # Fit all training data
        for epoch in range(training_epochs):
            # randomize = np.arange(len(train_X))
            # np.random.shuffle(randomize)
            # train_X = train_X[randomize]
            # train_Y = train_Y[randomize]

            start_pos = np.random.randint(len(train_X) - batch_size)
            batch_x = train_X[start_pos:start_pos+batch_size].reshape((batch_size, n_input))
            batch_y = train_Y[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

            if epoch > 1:

                pred_vals = []
                for i in range(len(pred_X)):
                    pred_val = sess.run(pred, feed_dict={x: pred_X[i].reshape((1,n_input)), keep_prob:1.0})
                    # print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(pred_Y[i])
                    pred_vals.append(pred_val[0][0])
                avg_pred_vals.append(np.array(pred_vals))

                slope, intercept, r_value, p_value, std_err = linregress(np.mean(np.array(avg_pred_vals), axis=0), pred_Y)

                # print "R^2 = " + str(r_value**2)
                # print "Slope = " + str(slope)
                # print "Intercept = " + str(intercept)
                # print "Epoch = " + str(epoch)
                # print "Loss = " + str(np.mean(np.absolute(loss)))
                # print "Accuracy = " + str(np.mean(np.absolute(acc)))
                # print "\n"

                predValsPerEpoch = []
                for i in range(len(pred_X)):
                    predValsPerEpoch.append(slope * np.mean(np.array(avg_pred_vals)[:,i] + intercept))
                #     print str(slope * np.mean(np.array(avg_pred_vals)[:,i]) + intercept) + "," + str(pred_Y[i])
                # print "\n"

                vals = []
                for i in range(predictionFile.shape[0]):
                    pred_val = sess.run(pred, feed_dict={x: predictionFile[i,:].reshape((1,n_input)), keep_prob: 1.0})
                    vals.append(float(pred_val))
                avgPredVals.append(vals)

                adjustedAvgPreds = []
                tempArr = np.array(avgPredVals)
                for i in range(tempArr.shape[1]):
                    adjustedAvgPreds.append(slope * np.mean(tempArr[:,i]) + intercept)
                #     print str(team_hash_table[int(predictionFile[i,0])][0]) + ",  " + str(team_hash_table[int(predictionFile[i,1])][0]) + ",  " + str(slope * np.mean(tempArr[:,i]) + intercept)
                # print "\n"

    overallAvgPreds.append(adjustedAvgPreds)
    tf.reset_default_graph()

    if k % 2 == 0:
        tempArr = np.array(overallAvgPreds)
        correctCount = 0
        correctGames = []
        for i in range(tempArr.shape[1]):
            print str(team_hash_table[int(predictionFile[i,0])][0]) + ",  " + str(team_hash_table[int(predictionFile[i,1])][0]) + ",  " + str(np.mean(tempArr[:,i]))
        print "\n"
        for i in range(tempArr.shape[1]):
            score = -1 * np.mean(tempArr[:,i])
            if (score > spreads[i,0] and spreads[i,1] > spreads[i,0]) or (score < spreads[i,0] and spreads[i,1] < spreads[i,0]):
                print str(spreads[i,0]) + "," + str(spreads[i,1]) + "," + str(score) + ",1"
                correctCount = correctCount + 1
                correctGames.append([spreads[i,0], spreads[i,1], score])
            else:
                print str(spreads[i,0]) + "," + str(spreads[i,1]) + "," + str(score) + ",0"

        print "\n"
        print "Number correct = " + str(correctCount)
        print "Accuracy = " + str(float(correctCount)/float(tempArr.shape[1]))

        absAvgDiff = []
        absMedianDiff = []
        for i in range(len(scores)):
            absAvgDiff.append(math.fabs(scores[i] - np.mean(tempArr[:,i])))
            absMedianDiff.append(math.fabs(scores[i] - np.median(tempArr[:,i])))
            avgPredDiff.append(scores[i] - np.mean(tempArr[:,i]))
        print "\n"
        print "Average mean difference = " + str(np.mean(absAvgDiff))
        print "Median mean difference = " + str(np.median(absAvgDiff))
        print "Average median difference = " + str(np.mean(absMedianDiff))
        print "Median median difference = " + str(np.median(absMedianDiff))
        print "Average difference (GT - Pred) = " + str(np.mean(avgPredDiff))
        print "\n"
