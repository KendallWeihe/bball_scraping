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
training_epochs = 300
display_step = 10
n_predictions = 0
batch_size = 50

predictionFile = np.genfromtxt(sys.argv[1], delimiter=",")

# Training Data
path = "./stats/w_score/"
files = os.listdir(path)
files.sort()
data = np.genfromtxt(path+files[0], delimiter=",")
for i in range(len(files)):
    if "stats/w_score/"+files[i] != sys.argv[1]:
        temp = np.genfromtxt(path+files[i], delimiter=",")
        data = np.concatenate((data, temp), axis=0)

vegasSpreads = data[:,4:6].copy()
actualScores = data[:,36:38].copy()
data = data[:,0:36]

# 0.5566
randomlyGeneratedStatColumns = [9, 30, 12, 29, 23, 13, 15, 20,  7,  5, 32, 35, 28, 33, 26, 27, 22, 11, 21, 17, 34, 14, 16,  6, 25, 31,  8, 19,  4,  1]
# 0.5632
randomlyGeneratedStatColumns = [15, 26, 12, 23, 17, 33,  1, 30, 32, 29, 16, 27, 10,  3, 20, 35, 14, 34, 21,  9,  6, 28, 13,  8, 19, 2, 22,  5]
data = np.take(data, randomlyGeneratedStatColumns, axis=1)

teams = predictionFile[:,0]
predictionSpreads = predictionFile[:,4].copy() - predictionFile[:,5].copy()
predictionScores = predictionFile[:,36].copy() - predictionFile[:,37].copy()
predictionFile = np.take(predictionFile, randomlyGeneratedStatColumns, axis=1)

n_input = data.shape[1]
n_classes = 1
dropout = 0.75

for i in range(data.shape[1]):
    predictionFile[:,i] = (predictionFile[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))
    data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.amax(data[:,i]) - np.min(data[:,i]))
    mean = np.mean(data[:,i])
    sigma = np.std(data[:,i])
    predictionFile[:,i] = (predictionFile[:,i] - mean) / sigma
    data[:,i] = (data[:,i] - mean) / sigma

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

randomize = np.arange(len(data))
np.random.shuffle(randomize)
data = data[randomize]
vegasSpreads = vegasSpreads[randomize]
actualScores = actualScores[randomize]

train_X = data[0:data.shape[0]-n_predictions,:]
train_Y = actualScores[0:actualScores.shape[0]-n_predictions,0] - actualScores[0:actualScores.shape[0]-n_predictions,1]
pred_X = data[data.shape[0]-n_predictions:data.shape[0],:]
pred_Y = actualScores[actualScores.shape[0]-n_predictions:actualScores.shape[0],0] - actualScores[actualScores.shape[0]-n_predictions:actualScores.shape[0],1]

overallAvgPreds = []
for k in range(25):

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Create model
    n_hidden = 128
    n_steps = 1
    n_neurons = 25
    def network(x):
        x = tf.reshape(x, [-1, n_input])
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
        avgPredictionValues = []
        # Fit all training data
        for epoch in range(training_epochs):
            start_pos = np.random.randint(len(train_X) - batch_size)
            batch_x = train_X[start_pos:start_pos+batch_size].reshape((batch_size, n_input))
            batch_y = train_Y[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            tempPredVals = []
            for i in range(predictionFile.shape[0]):
                predictionValue = float(sess.run(pred, feed_dict={x: predictionFile[i].reshape((1,n_input)), keep_prob:1.0}))
                tempPredVals.append(predictionValue)
            avgPredictionValues.append(tempPredVals)

        correctCount = 0
        gameCount = 0
        avgPredVals = np.mean(avgPredictionValues, axis=0)
        for i in range(avgPredVals.shape[0]):
            if (predictionSpreads[i] > avgPredVals[i] and predictionSpreads[i] > predictionScores[i]) or (predictionSpreads[i] < avgPredVals[i] and predictionSpreads[i] < predictionScores[i]):
                correctCount = correctCount + 1

        print "Average acc: " + str(float(correctCount)/float(avgPredVals.shape[0]))

    tf.reset_default_graph()
    overallAvgPreds.append(avgPredVals)

correctCount = 0
gameCount = 0
overallAvgPredVals = np.mean(overallAvgPreds, axis=0)
for i in range(overallAvgPredVals.shape[0]):
    # if math.fabs(predictionSpreads[i] - avgPredVals[i]) > 5:
    if (predictionSpreads[i] > overallAvgPredVals[i] and predictionSpreads[i] > predictionScores[i]) or (predictionSpreads[i] < overallAvgPredVals[i] and predictionSpreads[i] < predictionScores[i]):
        correctCount = correctCount + 1
print "\nFinal average accuracy: " + str(float(correctCount)/float(overallAvgPredVals.shape[0]))
