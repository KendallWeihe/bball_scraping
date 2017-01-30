#TODO invoke generateSecondaryData.py with new file name

import os
import pdb

files = os.listdir("./stats/w_score/")
for f in files:
    os.system("python generateSecondaryData.py stats/w_score/" + f + " " + f)
