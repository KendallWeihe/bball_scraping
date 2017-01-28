import numpy as np
import pdb
import math

results = np.genfromtxt("./ComparisonStats.csv", delimiter=",")
print "Total number of games: " + str(results.shape[0]) + "\n"

correctCount = 0
totalCount = 0
for i in range(results.shape[0]):
    if (results[i,0] > 0 and results[i,1] < 0) or (results[i,0] < 0 and results[i,1] > 0):
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctCount = correctCount + 1
        totalCount = totalCount + 1

print "Number of games vegas predicted wrong: " + str(totalCount)
print "Number of games I correctly predicted: " + str(correctCount)
print "Accuracy: " + str(float(correctCount)/float(totalCount))

correctGamesGreater3 = 0
numGamesGreater3 = 0
for i in range(results.shape[0]):
    if math.fabs(results[i,0] - results[i,2]) >= 3:
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctGamesGreater3 = correctGamesGreater3 + 1
        numGamesGreater3 = numGamesGreater3 + 1

print "\nNumber of games where difference > 3: " + str(numGamesGreater3)
print "Number of games correct > 3: " + str(correctGamesGreater3)

correctGamesGreater5 = 0
numGamesGreater5 = 0
for i in range(results.shape[0]):
    if math.fabs(results[i,0] - results[i,2]) >= 5:
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctGamesGreater5 = correctGamesGreater5 + 1
        numGamesGreater5 = numGamesGreater5 + 1

print "\nNumber of games where difference > 5: " + str(numGamesGreater5)
print "Number of games correct > 5: " + str(correctGamesGreater5)

correctGamesGreater7 = 0
numGamesGreater7 = 0
for i in range(results.shape[0]):
    if math.fabs(results[i,0] - results[i,2]) >= 7:
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctGamesGreater7 = correctGamesGreater7 + 1
        numGamesGreater7 = numGamesGreater7 + 1

print "\nNumber of games where difference > 7: " + str(numGamesGreater7)
print "Number of games correct > 7: " + str(correctGamesGreater7)

correctGamesGreater9 = 0
numGamesGreater9 = 0
for i in range(results.shape[0]):
    if math.fabs(results[i,0] - results[i,2]) >= 9:
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctGamesGreater9 = correctGamesGreater9 + 1
        numGamesGreater9 = numGamesGreater9 + 1

print "\nNumber of games where difference > 9: " + str(numGamesGreater9)
print "Number of games correct > 9: " + str(correctGamesGreater9)

correctGamesGreater11 = 0
numGamesGreater11 = 0
for i in range(results.shape[0]):
    if math.fabs(results[i,0] - results[i,2]) >= 11:
        if (results[i,1] > results[i,0] and results[i,2] > results[i,0]) or (results[i,1] < results[i,0] and results[i,2] < results[i,0]):
            correctGamesGreater11 = correctGamesGreater11 + 1
        numGamesGreater11 = numGamesGreater11 + 1

print "\nNumber of games where difference > 11: " + str(numGamesGreater11)
print "Number of games correct > 11: " + str(correctGamesGreater11)
