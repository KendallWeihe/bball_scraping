import numpy as np
import pdb
import glob
import os

files = glob.glob("./ncaa_data/*.csv")

# print "Number of files = " + str(len(files))
avg_shape = []
halftime_line = []
total_num_lines = []
error_count = 0
for csv_file in files:
    try:
        csv_data = np.genfromtxt(csv_file, delimiter=",")
        if csv_data.shape[0] > 160 and csv_data[0,0] < 3 and csv_data[csv_data.shape[0]-1,0] > 37:
            print csv_file
            os.system("mv " + csv_file + " ./ncaa_data/completed_games/")
        # if csv_data.shape[0] < 200:
        #     print "Less than 200 -- " + csv_file
        #     os.system("mv " + csv_file + " ./ncaa_data/garbage/")
        # elif csv_data[0,0] > 2:
        #     print "Greater than 2 -- " + csv_file
        #     os.system("mv " + csv_file + " ./ncaa_data/garbage/")
        # elif csv_data[csv_data.shape[0]-1,0] < 38:
        #     print "Less than 38 -- " + csv_file
        #     os.system("mv " + csv_file + " ./ncaa_data/garbage/")
        # avg_shape.append(csv_data.shape[0])
        # for i in range(csv_data.shape[0]):
        #     if csv_data[i,0] == 20 or csv_data[i,0] == 21:
        #         halftime_line.append(i)
        #         break
        total_num_lines.append(csv_data.shape[0])
    except:
        print "Error -- " + csv_file
        # pdb.set_trace()
        error_count = error_count + 1

print "Error count = " + str(error_count)
print "Number of files = " + str(len(files))
print "Average halftime line: " + str(np.mean(np.array(halftime_line))) # 127
print "Average number of lines = " + str(np.mean(np.array(total_num_lines)))

# print np.mean(np.array(avg_shape))
