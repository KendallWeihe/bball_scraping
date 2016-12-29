import pdb
import requests
from bs4 import BeautifulSoup
import sys
import difflib
import re
import numpy as np
import math
import time

# ARGUMENTS:
    # team_1
    # team_2
    # date

link = "http://www.covers.com/odds/basketball/college-basketball-2nd-half-lines.aspx"

time_count = 0
while time_count < 20:

    r = requests.get(link)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.findAll('tr', class_="bg_row")

    team_names = []
    for row in rows:
        row_bs4 = BeautifulSoup(str(row), "html.parser")
        td = row_bs4.findAll('td')
        team_names.append(re.sub(r"\r?\n?", "", str(td[0].text)).split("@")[0])
        team_names.append(re.sub(r"\r?\n?", "", str(td[0].text)).split("@")[1])

    if difflib.get_close_matches(sys.argv[1], team_names, n=1):
        team_index = np.argwhere(np.array(team_names) == difflib.get_close_matches(sys.argv[1], team_names, n=1))[0,0]
        actual_team_index = float(team_index)/2
        team_index = int(math.floor(team_index/2))
    else:
        sys.exit()

    game_row = rows[team_index]
    game_row = BeautifulSoup(str(game_row), "html.parser")
    td = game_row.findAll('td')

    spreads = []
    for i in range(3,len(td)):
        if re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", str(td[i].text)) != []:
            spreads.append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", str(td[i].text)))

    if spreads != []:
        try:
            spreads = np.array(spreads)[:,1]
            avg_spread = np.mean(spreads.astype(float))
        except:
            spreads = np.array(spreads)
        if actual_team_index % 1 != 0:
            my_spread = avg_spread * -1
        else:
            my_spread = avg_spread

        filename = "./half_time_spreads/" + sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + ".csv"
        arr = np.array([my_spread])
        np.savetxt(filename, arr, delimiter=",")

    time.sleep(60)
    time_count = time_count + 1
