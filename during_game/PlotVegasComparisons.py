import numpy as np
import pdb
import matplotlib.pyplot as plt
import math

# TODO
    # iterate through every game
    # sort game into respective vector positions
    #     based on rounded difference between my pred and vegas line

data = np.genfromtxt("./ComparisonsMine.csv", delimiter=",")

correct_preds = np.zeros(40)
total_preds = np.zeros(40)
for i in range(data.shape[0]):
    if data[i,3] == 1:
        correct_preds[math.fabs(data[i,0] - data[i,2])] = correct_preds[math.fabs(data[i,0] - data[i,2])] + 1
    total_preds[math.fabs(data[i,0] - data[i,2])] = total_preds[math.fabs(data[i,0] - data[i,2])] + 1

fig, axs = plt.subplots(2,1,figsize=(40,8))

x = np.arange(40)

axs[0].bar(x, correct_preds, width=0.4, facecolor='b', edgecolor='b', linewidth=3, alpha=.5)
axs[0].bar(x+0.4, total_preds, width=0.4, facecolor='r', edgecolor='r', linewidth=3, alpha=.5)

plt.plot()
# plt.show()

pdb.set_trace()
output = np.stack((correct_preds, total_preds), 1)
np.savetxt("./VegasRangeComparisons.csv", output, delimiter=",")
