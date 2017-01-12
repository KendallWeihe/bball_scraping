import numpy as np
import math

data = np.genfromtxt("./ComparisonsMine.csv", delimiter=",")

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 3:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print " > 3 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 3 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 5:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print " > 5 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 5 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 8:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print " > 8 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 8 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 12:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print " > 12 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 12 ==> " + str(overall_count)

correct_count = 0
overall_count = 0
for i in range(data.shape[0]):
    if math.fabs(data[i,0] - data[i,2]) > 15:
        if data[i,3] == 1:
            correct_count = correct_count + 1
        overall_count = overall_count + 1

print " > 15 ==> " + str(float(correct_count)/float(overall_count))
print "Number of games > 15 ==> " + str(overall_count)
