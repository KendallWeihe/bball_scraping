import numpy as np
import pdb
import glob

files = glob.glob("./ncaa_data/completed_games/*.csv")

print "Number of files = " + str(len(files))
avg_shape = []
halftime_line = []
for csv_file in files:
    csv_data = np.genfromtxt(csv_file, delimiter=",")
    if csv_data.shape[0] < 200:
        print "Less than 200 "
        print csv_file
    if csv_data[0,0] > 2:
        print "Greater than 2"
        print csv_file
    avg_shape.append(csv_data.shape[0])
    for i in range(csv_data.shape[0]):
        if csv_data[i,0] == 20 or csv_data[i,0] == 21:
            halftime_line.append(i)
            break

print "Halftime line: " + str(np.mean(np.array(halftime_line))) # 127

# print np.mean(np.array(avg_shape))