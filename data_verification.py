import numpy as np
import pdb
import glob

files = glob.glob("./ncaa_data/completed_games/*.csv")

avg_shape = []
print "Number of files = " + str(len(files))
for csv_file in files:
    csv_data = np.genfromtxt(csv_file, delimiter=",")
    # if csv_data.shape[0] < 200:
    #     print csv_file
    avg_shape.append(csv_data.shape[0])

print np.mean(np.array(avg_shape))
