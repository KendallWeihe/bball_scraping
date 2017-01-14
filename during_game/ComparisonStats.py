import numpy as np
import math

data = np.genfromtxt("./ComparisonsMine.csv", delimiter=",")

temp = data[:,3]
print "Overall accuracy ==> " + str(float(len(temp[temp > 0]))/float(data.shape[0]))
print "Total number of games ==> " + str(data.shape[0])

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 3:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print "Accuracy > 3 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 3 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 5:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print "Accuracy > 5 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 5 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 8:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print "Accuracy > 8 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 8 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 12:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print "Accuracy > 12 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 12 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 15:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print "Accuracy > 15 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 15 ==> " + str(overall_count)
