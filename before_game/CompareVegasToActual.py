import numpy as np
import requests
from bs4 import BeautifulSoup
import pdb
import sys
import math

link = "http://www.vegasinsider.com/college-basketball/scoreboard/scores.cfm/game_date/01-14-2017"

r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")

tables = soup.findAll('table')

differences = []
for table in tables:
    try:
        if len(table.findAll('td')) == 24:
            t = table.findAll('td')
            oddsLocation = False
            oddsTeam1 = float(t[12].string)
            if float(oddsTeam1) > 45:
                oddsTeam2 = float(t[18].string)
                oddsLocation = True
            score1 = float(t[15].string)
            score2 = float(t[21].string)

            if not oddsLocation:
                finalSpread = score2 - score1
                difference = math.fabs(finalSpread - oddsTeam1)
            else:
                finalSpread = score1 - score2
                difference = math.fabs(finalSpread - oddsTeam2)
            if difference < 45:
                print difference
                differences.append(difference)

    except:
        pass

print "Average difference = " + str(np.mean(differences))
