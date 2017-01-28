import numpy as np
import pdb
import csv
import os

# TODO:
    # open team hash table
    # open w_score files
    # for each file:
    #     print file name
    #     for each row:
    #         print team (from team hash table)
    #         ask for input
    #         append input to array
    #     save the file to w_spreads/ dir

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

path = "stats/w_score/"
files = os.listdir(path)
files.sort()
for f in files:
    print "File w/ date: " + f
    print "Skip? Y/N"
    if raw_input() != "Y":
        data = np.genfromtxt(path + f, delimiter=",")
        newData = []
        for row in data:
            print "Team: " + team_hash_table[int(row[0])][0]
            spread = float(raw_input("Enter spread: "))
            newData.append(np.append(row, spread))
            print "\n"
        np.savetxt("./WithSpreads/"+f, np.array(newData), delimiter=",")
