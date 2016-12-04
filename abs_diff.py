import numpy as np
import pdb
import math

data = np.genfromtxt("temp.csv", delimiter=",")

# pdb.set_trace()

preds = []
actuals = []
count = 0
for i in range(0,data.shape[0],2):
    preds.append(data[i,0])
    actuals.append(data[i,1])
    if math.fabs(data[i,0] - data[i,1]) < 6:
        count = count + 1

print "Average difference = " + str(np.mean(np.absolute(np.array(preds)-np.array(actuals))))
print "Median difference = " + str(np.median(np.absolute(np.array(preds)-np.array(actuals))))
print "Total number of games == " + str(data.shape[0])
print "Number of games differences < 6 == " + str(count)
