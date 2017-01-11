import numpy as np
import pdb
import math
import glob

data = np.genfromtxt("regression.csv", delimiter=",")

# pdb.set_trace()

preds = []
actuals = []
count = 0
win_count = 0
for i in range(0,data.shape[0],2):
    preds.append(data[i,0])
    actuals.append(data[i,1])
    if math.fabs(data[i,0] - data[i,1]) < 6:
        count = count + 1
    if (data[i,0] > 0 and data[i,1] > 0) or (data[i,0] < 0 and data[i,1] < 0):
         win_count = win_count + 1

print "Average difference = " + str(np.mean(np.absolute(np.array(preds)-np.array(actuals))))
print "Median difference = " + str(np.median(np.absolute(np.array(preds)-np.array(actuals))))
print "Total number of games == " + str(data.shape[0])
print "Number of games differences < 6 == " + str(count)
print "Number of correct games = " + str(win_count)

def input_data():
    files = glob.glob("./ncaa_data/completed_games/*.csv")
    print "Number of files = " + str(len(files))

    avg_half_time = []
    avg_time_at_step = []
    for csv_file in files:
        try:
            csv_data = np.genfromtxt(csv_file, delimiter=",")
            for i in range(csv_data.shape[0]):
                if csv_data[i,0] == 20 or csv_data[i,0] == 21:
                    avg_half_time.append(i)
                    break
            avg_time_at_step.append(csv_data[175,0])
        except:
            print csv_file

    return np.mean(np.array(avg_half_time)), np.mean(np.array(avg_time_at_step))

avg_half_time, avg_time_at_step = input_data()

print "Average halftime line = " + str(avg_half_time)
print "Average time at step 175 = " + str(avg_time_at_step)
