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
#     prediction file name and path -- located in stats/w_out_score/

if len(sys.argv) != 2:
    print "Missing arguments"

# Parameters
learning_rate = 0.01
training_epochs = 750
display_step = 10
n_predictions = 50
batch_size = 50

# Training Data
path = "./stats/w_score/"
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

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

overallAvgPreds = []
predValsPerSession = []
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
    n_hidden_1 = 2500 # 1st layer number of features
    n_hidden_2 = 500 # 2nd layer number of features

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.l2_normalize(layer_1, 1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        return tf.nn.dropout(layer_2, 0.75)
        # layer_2 = tf.nn.relu(layer_2)
        # layer_2 = tf.nn.l2_normalize(layer_2, 1)

        # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        # return tf.nn.dropout(out_layer, 0.75)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_classes])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_classes])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Construct model
    n_samples = tf.cast(tf.shape(x)[0], tf.float32)
    cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    accuracy = tf.abs(tf.sub(pred, y))
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
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

            pred_vals = []
            for i in range(len(pred_X)):
                pred_val = sess.run(pred, feed_dict={x: pred_X[i].reshape((1,n_input))})
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

            # actualValues = []
            # for i in range(len(pred_X)):
            #     actualValues.append(np.array(avg_pred_vals)[:,i][0])
                # print str(np.array(avg_pred_vals)[:,i]) + "," + str(pred_Y[i])
            # print "\n"
            # -------------------------------------------------------------------------------------
            # differences = []
            # for i in range(len(actualValues)):
            #     differences.append(math.fabs(actualValues[i] - pred_Y[i]))
            # upper10Percentile = np.percentile(differences, 90)
            #
            # adjustedPoints = []
            # adjustedPointsGT = []
            # for i in range(len(actualValues)):
            #     if differences[i] < upper10Percentile:
            #         adjustedPoints.append(actualValues[i])
            #         adjustedPointsGT.append(pred_Y[i])
            #
            # slope, intercept, r_value, p_value, std_err = linregress(adjustedPoints, adjustedPointsGT)
            # print "R^2 = " + str(r_value**2)
            # print "Slope = " + str(slope)
            # print "Intercept = " + str(intercept)

            # for i in range(len(adjustedPoints)):
            #     print str(slope * adjustedPoints[i] + intercept) + "," + str(adjustedPointsGT[i])
            # print "\n"
            # -------------------------------------------------------------------------------------
            vals = []
            for i in range(predictionFile.shape[0]):
                pred_val = sess.run(pred, feed_dict={x: predictionFile[i,:].reshape((1,n_input))})
                vals.append(pred_val)
            avgPredVals.append(vals)

            adjustedAvgPreds = []
            tempArr = np.array(avgPredVals)
            for i in range(tempArr.shape[1]):
                adjustedAvgPreds.append(slope * np.mean(tempArr[:,i]) + intercept)
            #     print str(team_hash_table[int(predictionFile[i,0])][0]) + ",  " + str(team_hash_table[int(predictionFile[i,1])][0]) + ",  " + str(slope * np.mean(tempArr[:,i]) + intercept)
            # print "\n"

    overallAvgPreds.append(adjustedAvgPreds)
    predValsPerSession.append(predValsPerEpoch)
    tf.reset_default_graph()

    if k % 2 == 0:
        tempArr = np.array(overallAvgPreds)
        for i in range(tempArr.shape[1]):
            print str(team_hash_table[int(predictionFile[i,0])][0]) + ",  " + str(team_hash_table[int(predictionFile[i,1])][0]) + ",  " + str(np.mean(tempArr[:,i]))
        groundTruth = [-5,10,-2,-19,-10,9,-6,-16,9,6,-19]
        tempArr = np.delete(tempArr, 2, 1)
        tempArr = np.delete(tempArr, 0, 1)
        absAvgDiff = []
        absMedianDiff = []
        for i in range(len(groundTruth)):
            absAvgDiff.append(math.fabs(groundTruth[i] - np.mean(tempArr[:,i])))
            absMedianDiff.append(math.fabs(groundTruth[i] - np.median(tempArr[:,i])))
            avgPredDiff.append(groundTruth[i] - np.mean(tempArr[:,i]))
        print "\n"
        print "Average mean difference = " + str(np.mean(absAvgDiff))
        print "Median mean difference = " + str(np.median(absAvgDiff))
        print "Average median difference = " + str(np.mean(absMedianDiff))
        print "Median median difference = " + str(np.median(absMedianDiff))
        print "Average difference (GT - Pred) = " + str(np.mean(avgPredDiff))
        print "\n"

# tempArr = np.array(predValsPerSession)
# for i in range(tempArr.shape[1]):
#     print str(np.mean(tempArr[:,i])) + "," + str(pred_Y[i])
