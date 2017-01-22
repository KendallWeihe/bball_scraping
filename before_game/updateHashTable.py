import csv
import numpy as np
import requests
from bs4 import BeautifulSoup
import pdb
import sys

# TODO:
    # also record the hash table for each day

# ARGV:
#    link
#     output file name

team_hash_table = []
with open('team_hash_table.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

link = sys.argv[1]
r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")

rows = soup.findAll('tr')

outputHash = []
# for i in range(351):

for i in range(1,352):
    teamName = str(rows[i].findAll('a')[0].string)
    for j in range(len(team_hash_table)):
        if team_hash_table[j][0].replace("_", " ") == teamName:
            # print teamName.replace(" ", "_") + "," + team_hash_table[j][1]
            outputHash.append([teamName.replace(" ", "_"), team_hash_table[j][1]])


with open(sys.argv[2], 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in outputHash:
        writer.writerow(row)
