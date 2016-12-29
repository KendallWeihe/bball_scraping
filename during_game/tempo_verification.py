import numpy as np
import os
import pdb

# TODO
    # open all files
    # iterate through each file
    #     iterate through each row
    #         if [i+1] - [i] > 2:
    #             move file


files = os.listdir("./ncaa_data/")
print "Number of files: " + str(len(files))

for f in files:
    try:
        game = np.genfromtxt("./ncaa_data/" + f, delimiter=",")
        for i in range(game.shape[0]-1):
            if (game[i+1,0] - game[i,0]) > 1:
                os.system("mv " + "./ncaa_data/" + f + " ./ncaa_data/clean_tempo/")
    except:
        print "Error: " + f
