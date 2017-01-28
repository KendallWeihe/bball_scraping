import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
from tensorflow.python.ops import rnn, rnn_cell
import tensorflow.contrib.slim as slim
import os
from scipy.stats import linregress
import math
import sys
import csv

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

# Parameters
learning_rate = 0.01
dropout = 0.75
training_epochs = 300
display_step = 10
n_predictions = 50
batch_size = 50
n_input = 36
n_classes = 1

# Training Data
path = "stats/w_score/"
files = os.listdir(path)
files.sort()
data = np.genfromtxt(path+files[0], delimiter=",")
for i in range(len(files)):
    temp = np.genfromtxt(path+files[i], delimiter=",")
    data = np.concatenate((data, temp), axis=0)

vegasSpreads = data[:,4:6].copy()
actualScores = data[:,36:38].copy()
data = data[:,0:36]

for i in range(data.shape[1]-2):
    data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))
    mean = np.mean(data[:,i])
    sigma = np.std(data[:,i])
    data[:,i] = (data[:,i] - mean) / sigma

def compareWithVegas(predictionSpreads, predictionScores, actualScores):
    correctCount = 0
    for i in range(predictionScores.shape[0]):
        vegasSpread = predictionSpreads[i,0] - predictionSpreads[i,1]
        myPredictedSpread = predictionScores[i]
        actualSpread = actualScores[i,0] - actualScores[i,1]
        if (actualSpread > vegasSpread and myPredictedSpread > vegasSpread) or (actualSpread < vegasSpread and myPredictedSpread < vegasSpread):
            correctCount = correctCount + 1
    return float(correctCount) / 50.0

maxAcc = 0.0
maxAccStats = []
minAcc = 1.0
minAccStats = []
for l in range(100000):
    randomIndices = np.arange(n_input)
    np.random.shuffle(randomIndices)
    randomNumStats = np.random.randint(n_input)
    randomlyGeneratedStatColumns = randomIndices[0:randomNumStats]
    randomlySelectedData = np.take(data, randomlyGeneratedStatColumns, axis=1)

    overallAvgPreds = []
    avgPredDiff = []
    avgVegasAccuracy = []
    for k in range(25):

        randomize = np.arange(len(data))
        np.random.shuffle(randomize)
        data = data[randomize]
        vegasSpreads = vegasSpreads[randomize]
        actualScores = actualScores[randomize]

        train_X = randomlySelectedData[0:randomlySelectedData.shape[0]-n_predictions,:]
        train_Y = actualScores[0:actualScores.shape[0]-n_predictions,0] - actualScores[0:actualScores.shape[0]-n_predictions,1]

        pred_X = randomlySelectedData[randomlySelectedData.shape[0]-n_predictions:randomlySelectedData.shape[0],:]
        pred_Y = actualScores[actualScores.shape[0]-n_predictions:actualScores.shape[0],0] - actualScores[actualScores.shape[0]-n_predictions:actualScores.shape[0],1]

        predictionSpreads = vegasSpreads[vegasSpreads.shape[0]-n_predictions:vegasSpreads.shape[0],:]
        predictionSetScores = actualScores[actualScores.shape[0]-n_predictions:actualScores.shape[0],:]

        # tf Graph input
        x = tf.placeholder("float", [None, randomNumStats])
        y = tf.placeholder("float", [None, n_classes])
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # Create model
        n_hidden = 128
        n_steps = 1
        n_neurons = 25
        def network(x):
            x = tf.reshape(x, [-1, randomNumStats])
            x = tf.split(0, n_steps, x)
            lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
            outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
            output = tf.matmul(outputs[-1], weights['out']) + biases['out']
            return tf.nn.dropout(output, dropout)

        # Store layers weight & bias
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        pred = network(x)

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
            adjustedAvgPreds = []
            # Fit all training data
            for epoch in range(training_epochs):
                start_pos = np.random.randint(len(train_X) - batch_size)
                batch_x = train_X[start_pos:start_pos+batch_size].reshape((batch_size, randomNumStats))
                batch_y = train_Y[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

                pred_vals = []
                for i in range(len(pred_X)):
                    pred_val = sess.run(pred, feed_dict={x: pred_X[i].reshape((1,randomNumStats)), keep_prob:1.0})
                    pred_vals.append(float(pred_val))
                avg_pred_vals.append(np.array(pred_vals))

                slope, intercept, r_value, p_value, std_err = linregress(np.mean(np.array(avg_pred_vals), axis=0), pred_Y)

                # print "R^2 = " + str(r_value**2)
                # print "Slope = " + str(slope)
                # print "Intercept = " + str(intercept)
                # print "Epoch = " + str(epoch)
                # print "Loss = " + str(np.mean(np.absolute(loss)))
                # print "Accuracy = " + str(np.mean(np.absolute(acc)))
                # print "\n"

                adjustedPreds = []
                tempArr = np.array(avg_pred_vals)
                for i in range(tempArr.shape[1]):
                    adjustedPreds.append(slope * np.mean(tempArr[:,i]) + intercept)
                adjustedAvgPreds.append(adjustedPreds)

        acc = compareWithVegas(predictionSpreads, np.mean(adjustedAvgPreds, axis=0), predictionSetScores)
        avgVegasAccuracy.append(acc)
        print "Average accuracy against vegas: " + str(np.mean(avgVegasAccuracy))
        tf.reset_default_graph()

    if np.mean(avgVegasAccuracy) > maxAcc:
        maxAcc = np.mean(avgVegasAccuracy)
        maxAccStats = randomlyGeneratedStatColumns
    if np.mean(avgVegasAccuracy) < minAcc:
        minAcc = np.mean(avgVegasAccuracy)
        minAccStats = randomlyGeneratedStatColumns

    print "Current maximum accuracy: " + str(maxAcc)
    print "Current maximum accuracy based on stats: " + str(maxAccStats)
    print "Current minimum accuracy: " + str(minAcc)
    print "Current minimum accuracy based on stats: " + str(minAccStats)
