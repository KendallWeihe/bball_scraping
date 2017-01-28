import requests
from bs4 import BeautifulSoup
import pdb
from selenium import webdriver
import unicodedata
import time
from difflib import SequenceMatcher
import sys
import smtplib

# ARGV:
    # team1
    # team2
    # date

#TODO:
    # train and predict
    # get bovada spread
    # send text message

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import glob
from scipy.stats import linregress

predictionFile = sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".csv"
predictionGame = np.genfromtxt("./ncaa_data/" + predictionFile, delimiter=",")
n_steps = predictionGame.shape[0]

learning_rate = 0.001
training_iters = 100000
batch_size = 50
display_step = 5

n_input = 22
n_hidden = 50
n_classes = 1
n_predictions = 50

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    # 'all_out' : tf.get_variable("weights_1", shape=[n_steps*n_hidden, n_classes],
    #            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
    'out' : tf.get_variable("weights_1", shape=[n_hidden, n_classes],
               initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
}

biases = {
    'out': tf.Variable(tf.zeros([n_classes]))
}


def input_data():
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    # print "Number of files = " + str(len(files))

    input_data = []
    ground_truth = []
    scores = []
    for csv_file in files:
        try:
            csv_data = np.genfromtxt(csv_file, delimiter=",")
            if csv_data.shape[0] > n_steps and csv_file != "./ncaa_data/" + predictionFile:
                input_data.append(csv_data[0:n_steps,:])
                ground_truth.append(csv_data[csv_data.shape[0]-1,2] - csv_data[csv_data.shape[0]-1,3])
                scores.append(csv_data[n_steps,2] - csv_data[n_steps,3])
        except:
            # print csv_file
            pass

    return np.array(input_data), np.array(ground_truth), np.array(scores)

input_data, ground_truth, scores = input_data()
# for i in range(input_data.shape[2]):
#     input_data[:,:,i] = (input_data[:,:,i] - np.min(input_data[:,:,i])) / (np.amax(input_data[:,:,i]) - np.min(input_data[:,:,i]))
#     predictionGame[:,i] = (predictionGame[:,i] - np.min(input_data[:,:,i])) / (np.amax(input_data[:,:,i]) - np.min(input_data[:,:,i]))
randomize = np.arange(len(input_data))
np.random.shuffle(randomize)
input_data = input_data[randomize]
ground_truth = ground_truth[randomize]

training_data = input_data[0:input_data.shape[0] - n_predictions, :]
training_ground_truth = ground_truth[0:ground_truth.shape[0] - n_predictions]
prediction_data = input_data[input_data.shape[0] - n_predictions:input_data.shape[0], :]
prediction_ground_truth = ground_truth[ground_truth.shape[0] - n_predictions:ground_truth.shape[0]]

def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    all_lstm_outputs = tf.reshape(tf.stack(outputs, axis=1), [-1, n_steps*n_hidden])
    output = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # output = tf.matmul(all_lstm_outputs, weights['all_out']) + biases['out']
    return tf.nn.dropout(output, 0.75)

pred = RNN(x, weights, biases)
n_samples = tf.cast(tf.shape(x)[0], tf.float32)
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n_samples)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.abs(tf.sub(pred, y)))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

print "Predicting " + sys.argv[1] + " " + sys.argv[2]
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # saver.restore(sess, "./lstm_models/200/bball_scraping/lstm_models/200/lstm_model_n_steps_175_750_0.773770080762_1.33473893366_0.408350682807.ckpt")
    avg_pred_diff = []
    avg_pred_vals = []
    avg_reg_vals = []
    max_r_values = [0,0,0]
    accuracy_data = []
    single_game_pred = []
    for step in range(250):

        start_pos = np.random.randint(len(training_data) - batch_size)
        batch_x = training_data[start_pos:start_pos+batch_size].reshape((batch_size, n_steps, n_input))
        batch_y = training_ground_truth[start_pos:start_pos+batch_size].reshape((batch_size,n_classes))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        accuracy_data.append(np.mean(acc))
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

        if step > 1:
            pred_vals = []
            for i in range(prediction_data.shape[0]):
                pred_val = sess.run(pred, feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
                pred_vals.append(int(pred_val))
                avg_pred_diff.append(int(pred_val)-prediction_ground_truth[i])

            slope, intercept, r_value, p_value, std_err = linregress(np.array(pred_vals), prediction_ground_truth)

            avg_pred_vals.append(pred_vals)
            avg_pred_vals_np = np.array(avg_pred_vals)
            avg_vals = []
            for i in range(avg_pred_vals_np.shape[1]):
                avg_vals.append(np.mean(avg_pred_vals_np[:,i]))

            slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), prediction_ground_truth)
            max_r_values.append(r_value**2)
            if len(max_r_values) > 8:
                del max_r_values[np.where(np.array(max_r_values) == np.min(np.array(max_r_values)))[0][0]]

            for i in range(len(avg_vals)):
                actual_value = slope * avg_vals[i] + intercept

            pred_val = sess.run(pred, feed_dict={x: predictionGame.reshape((1,n_steps,n_input))})
            single_game_pred.append(pred_val)
            actualPrediction = slope * np.mean(np.array(single_game_pred)) + intercept
            # print "Actual prediction: " + str(actualPrediction)
            step += 1

tf.reset_default_graph()

def sendTextMessage(actualSpread, actualPrediction, team):
    gmail_user = 'kendallweihe@gmail.com'
    gmail_password = 'B0bbleh3adjoe'
    message = "\nTeam: " + team + " Network Prediction: " + str(actualPrediction)
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail("kendallweihe@gmail.com", "5022165761@vtext.com", message)
    except:
        print 'Something went wrong...'

sendTextMessage(1000, actualPrediction, sys.argv[1])

# gameNotFound = True
# timeCount = 0
# while gameNotFound and timeCount < 20:
#     link = "https://sports.bovada.lv/basketball/college-basketball"
#     driver = webdriver.Chrome("/home/kendall/Development/bball_scraping/during_game/chromedriver")
#     driver.get(link)
#     for i in range(5):
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(0.5)
#
#     soup = BeautifulSoup(driver.page_source, "html.parser")
#
#     games = soup.findAll('article')
#     for game in games:
#         try:
#             team1 = unicodedata.normalize('NFKD', game.findAll('h3')[0].string).encode('ascii','ignore')
#             team2 = unicodedata.normalize('NFKD', game.findAll('h3')[1].string).encode('ascii','ignore')
#             if SequenceMatcher(None, team1, sys.argv[1]).ratio() > 0.5 or SequenceMatcher(None, team1, sys.argv[2]).ratio() > 0.5:
#                 time = game.findAll('time')[0].string
#                 if "SECOND HALF" in time or "2H" in time:
#                     gameNotFound = False
#                     if SequenceMatcher(None, team1, sys.argv[1]).ratio() > 0.5:
#                         spread = game.findAll('ul')[2].findAll('li')[0].findAll('span')[0].string
#                         integerSpread = ""
#                         digitFound = False
#                         for i in range(len(spread)):
#                             if spread[i].isdigit():
#                                 integerSpread = integerSpread + str(spread[i])
#                                 digitFound = True
#                             if digitFound and not spread[i].isdigit():
#                                 fraction = unicodedata.numeric(spread[i])
#                         actualSpread = float(integerSpread) + float(fraction)
#                         actualPrediction = -1 * actualPrediction
#                         sendTextMessage(actualSpread, actualPrediction, sys.argv[1])
#                     elif SequenceMatcher(None, team1, sys.argv[2]).ratio() > 0.5:
#                         spread = float(game.findAll('ul')[2].findAll('li')[0].findAll('span')[1].string)
#                         integerSpread = ""
#                         digitFound = False
#                         for i in range(len(spread)):
#                             if spread[i].isdigit():
#                                 integerSpread = integerSpread + str(spread[i])
#                                 digitFound = True
#                             if digitFound and not spread[i].isdigit():
#                                 fraction = unicodedata.numeric(spread[i])
#                         actualSpread = float(integerSpread) + float(fraction)
#                         sendTextMessage(actualSpread, actualPrediction, sys.argv[1])
#
#         except:
#             pass
#
#     driver.quit()
#     time.sleep(60.0)
#     timeCount = timeCount + 1
