import numpy as np
import pdb
import os

# TODO
    # read in all the files from the half time spread
    # invoke predict_and_compare with the file name as the argument

files = os.listdir("./half_time_spreads/")

for f in files:
    os.system("python predict_and_compare.py " + f)
