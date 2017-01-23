# TODO
    # open all half time spread files
    # for halfTimeFile in files:
    #     TRY: use business logic from CompareVegasToActual.py to find the actual 2nd half spread
    #         open data file (with same name in different directory)
    #         find the number of steps
    #         train for 500
    #         train and predict for the next 250
    #         print spread, actual, mine, right/wrong

import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import pdb
from selenium import webdriver
import math
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import glob
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pyvirtualdisplay import Display

halfTimeSpreads = os.listdir("./half_time_spreads/")
espnLinks = []
for i in range(20161229,20161232):
    espnLinks.append("http://www.espn.com/mens-college-basketball/scoreboard/_/group/50/date/" + str(i))
for i in range(20170101,20170118):
    espnLinks.append("http://www.espn.com/mens-college-basketball/scoreboard/_/group/50/date/" + str(i))

def generateTeamNames(n, spreadFile):
    teams1 = []
    teams2 = []
    splitIndices = []
    for i in range(n):
        team1 = ""
        team2 = ""
        secondTeam = False
        for j in range(len(spreadFile)-11):
            if not secondTeam:
                if spreadFile[j] == "_":
                    if j in splitIndices:
                        team1 = team1 + " "
                    else:
                        splitIndices.append(j)
                        secondTeam = True
                else:
                    team1 = team1 + spreadFile[j]
            else:
                if spreadFile[j] == "_":
                    team2 = team2 + " "
                else:
                    team2 = team2 + spreadFile[j]
        teams1.append(team1)
        teams2.append(team2)
    return teams1, teams2

def find2H(link, teams1, teams2):
    r = requests.get(link)

    display = Display(visible=0, size=(800, 600))
    display.start()
    driver = webdriver.Chrome("/home/kendall/Development/bball_scraping/during_game/chromedriver")
    try:
        driver.get(link)

        time.sleep(.3)
        driver.find_element_by_css_selector("#scoreboard-page header .dropdown-type-group button").click()
        time.sleep(.3)
        driver.find_element_by_link_text('NCAA Division I').click()
        time.sleep(.3)

        soup = BeautifulSoup(driver.page_source)
        games = soup.find("div", {"id": "events"})

        for game in games:
            team1 = str(game.find_all("span", {"class": "sb-team-short"})[0].string.encode('utf-8'))
            team2 = str(game.find_all("span", {"class": "sb-team-short"})[1].string.encode('utf-8'))

            if team1 in teams1 and team2 in teams2:
                team1HalftimeScore = float(game.find_all("td")[1].string.encode("utf-8"))
                team2HalftimeScore = float(game.find_all("td")[5].string.encode("utf-8"))
                halftimeSpread = team1HalftimeScore - team2HalftimeScore

                team1FinalScore = float(game.find_all("td")[3].string.encode("utf-8"))
                team2FinalScore = float(game.find_all("td")[7].string.encode("utf-8"))
                finalSpread = team1FinalScore - team2FinalScore

                secondHalfSpread = finalSpread - halftimeSpread
                driver.quit()
                display.stop()
                return secondHalfSpread

        driver.quit()
        display.stop()
        return -1000
    except:
        driver.quit()
        display.stop()
        return -1000

count = 0
for spreadFile in halfTimeSpreads:
    print "Predicting game #" + str(count)
    count = count + 1
    vegas2HSpread = float(np.genfromtxt("./half_time_spreads/" + spreadFile, delimiter=","))
    try:
        n = 0
        for i in range(len(spreadFile)-11):
            if spreadFile[i] == "_":
                n = n + 1

        teams1, teams2 = generateTeamNames(n, spreadFile)

        spreadFound = False
        # pdb.set_trace()
        date = spreadFile[-10:-4]
        link = "http://www.espn.com/mens-college-basketball/scoreboard/_/date/20" + date
        actualSecondHalfSpread = find2H(link, teams1, teams2)
        if actualSecondHalfSpread == -1000:
            for link in espnLinks:
                actualSecondHalfSpread = find2H(link, teams1, teams2)
                if actualSecondHalfSpread > -1000:
                    spreadFound = True
                    break
        else:
            spreadFound = True

        if spreadFound:
            actualSecondHalfSpread = actualSecondHalfSpread * -1

            predictionGame = np.genfromtxt("./ncaa_data/completed_games/" + spreadFile, delimiter=",")
            for i in range(predictionGame.shape[0]):
                if predictionGame[i,0] > 19:
                    n_steps = i
                    break
            predictionGame = predictionGame[0:n_steps,:]

            learning_rate = 0.01
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
                    if csv_file != "./ncaa_data/completed_games/" + spreadFile:
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

            for i in range(input_data.shape[2]):
                input_data[:,:,i] = (input_data[:,:,i] - np.min(input_data[:,:,i])) / (np.amax(input_data[:,:,i]) - np.min(input_data[:,:,i]))
                predictionGame[:,i] = (predictionGame[:,i] - np.min(input_data[:,:,i])) / (np.amax(input_data[:,:,i]) - np.min(input_data[:,:,i]))

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

            # Launch the graph
            with tf.Session() as sess:
                sess.run(init)
                step = 1
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
                    # print "Step: " + str(step) + "  Accuracy: " + str(float(acc)) + "  Loss: " + str(loss)
                    # print "Step: " + str(step) + "  Mean Accuracy: " + str(np.mean(accuracy_data)) + "  Loss: " + str(loss)

                    if step > 1:
                        # print "Step = " + str(step)
                        # print "Accuracy = " + str(np.mean(acc))

                        pred_vals = []
                        for i in range(prediction_data.shape[0]):
                            pred_val = sess.run(pred, feed_dict={x: prediction_data[i].reshape((1,n_steps,n_input))})
                            pred_vals.append(int(pred_val))
                            avg_pred_diff.append(int(pred_val)-prediction_ground_truth[i])
                            # print "Prediction = " + str(pred_val[0][0]) + "  Actual = " + str(prediction_ground_truth[i])
                        # print "Prediction average = " + str(np.mean(np.abs(np.array(avg_pred_diff))))

                        slope, intercept, r_value, p_value, std_err = linregress(np.array(pred_vals), prediction_ground_truth)

                        avg_pred_vals.append(pred_vals)
                        avg_pred_vals_np = np.array(avg_pred_vals)
                        avg_vals = []
                        # print "Raw prediction data ----------------------------------------------"
                        for i in range(avg_pred_vals_np.shape[1]):
                            avg_vals.append(np.mean(avg_pred_vals_np[:,i]))
                            # print str(np.mean(avg_pred_vals_np[:,i])) + "," + str(prediction_ground_truth[i])
                            # print "Average for game " + str(i) + " = " + str(np.mean(avg_pred_vals_np[:,i])) + "  Actual = " + str(prediction_ground_truth[i])
                        # print "Slope before average = " + str(slope)
                        # print "R^2 before average = " + str(r_value**2)

                        slope, intercept, r_value, p_value, std_err = linregress(np.array(avg_vals), prediction_ground_truth)
                        # print "Slope after average = " + str(slope)
                        # print "R^2 after average = " + str(r_value**2)
                        # print "Intercept after average = " + str(intercept)
                        max_r_values.append(r_value**2)
                        if len(max_r_values) > 8:
                            del max_r_values[np.where(np.array(max_r_values) == np.min(np.array(max_r_values)))[0][0]]

                        # print "Actual prediction values --------------------------------------------"
                        for i in range(len(avg_vals)):
                            actual_value = slope * avg_vals[i] + intercept
                        #     print str(actual_value) + "," + str(prediction_ground_truth[i])
                        #
                        # print "\n"

                        step += 1

                        pred_val = sess.run(pred, feed_dict={x: predictionGame.reshape((1,n_steps,n_input))})
                        # print "Single game raw prediction = " + str(pred_val)
                        single_game_pred.append(pred_val)
                        # print "Single game average raw prediction = " + str(np.mean(np.array(single_game_pred)))
                        # print "Single game average actual prediction = " + str(slope * np.mean(np.array(single_game_pred)) + intercept)

                        myPrediction = slope * np.mean(np.array(single_game_pred)) + intercept

            myPrediction = myPrediction * -1
            if (actualSecondHalfSpread > vegas2HSpread and myPrediction > vegas2HSpread) or (actualSecondHalfSpread < vegas2HSpread and myPrediction < vegas2HSpread):
                comparisons = np.genfromtxt("./ComparisonsMine.csv", delimiter=",")
                if comparisons.size == 0:
                    comparisons = np.array([[vegas2HSpread, actualSecondHalfSpread, myPrediction, 1]])
                else:
                    comparisons = np.vstack((comparisons, [vegas2HSpread, actualSecondHalfSpread, myPrediction, 1]))
                np.savetxt("./ComparisonsMine.csv", comparisons, delimiter=",")
                print str(vegas2HSpread) + "," + str(actualSecondHalfSpread) + "," + str(myPrediction) + ",1"
            else:
                comparisons = np.genfromtxt("./ComparisonsMine.csv", delimiter=",")
                if comparisons.size == 0:
                    comparisons = np.array([[vegas2HSpread, actualSecondHalfSpread, myPrediction, 0]])
                else:
                    comparisons = np.vstack((comparisons, [vegas2HSpread, actualSecondHalfSpread, myPrediction, 0]))
                np.savetxt("./ComparisonsMine.csv", comparisons, delimiter=",")
                print str(vegas2HSpread) + "," + str(actualSecondHalfSpread) + "," + str(myPrediction) + ",0"
            tf.reset_default_graph()


    except:
        print "Error: " + spreadFile
