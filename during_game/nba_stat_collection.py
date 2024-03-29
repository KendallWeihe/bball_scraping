import pdb
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import sys
import time


# link = "http://www.espn.com/nba/matchup?gameId=400899645"
link = sys.argv[1]
over = False
output_table = []

while not over:

    r = requests.get(link)
    soup = BeautifulSoup(r.text, "html.parser")

    game_time = soup.find('span', class_="game-time")
    game_time = str(game_time.string)

    # pdb.set_trace(

    if game_time == "Final/OT" or game_time == "Final":
        over = True

    elif game_time != "Halftime" and game_time != 'None':
        # pdb.set_trace()
        try:
            table = soup.find('table', class_="mod-data")
            if table != None:
                team_rows = table.findAll('tr')

                score_1 = str(soup.find('div', class_="score icon-font-after").string)
                score_2 = str(soup.find('div', class_="score icon-font-before").string)

                field_goal_percentages = re.findall("\d+\.\d+", str(team_rows[2]))
                three_point_percentages = re.findall("\d+\.\d+", str(team_rows[4]))
                free_throw_percentages = re.findall("\d+\.\d+", str(team_rows[6]))
                total_rebounds = re.findall("\d+", str(team_rows[7]))
                assists = re.findall("\d+", str(team_rows[11]))
                steals = re.findall("\d+", str(team_rows[12]))
                blocks = re.findall("\d+", str(team_rows[13]))
                turnovers = re.findall("\d+", str(team_rows[14]))
                fast_break_points = re.findall("\d+", str(team_rows[16]))
                points_in_paint = re.findall("\d+", str(team_rows[17]))
                personal_fouls = re.findall("\d+", str(team_rows[18]))

                current_time = re.findall("\d+:\d+", game_time)[0]
                (minute, second) = current_time.split(":")
                if re.findall("nd", game_time) != []:
                    minute = (19 - float(minute)) + 20
                else:
                    minute = 19 - float(minute)
                second = 60 - float(second)

                row = np.array([[minute, second], [score_1, score_2], field_goal_percentages, three_point_percentages, free_throw_percentages, total_rebounds, assists, steals, blocks, turnovers, personal_fouls], dtype=np.float16)
                row = row.flatten()
                output_table.append(row)

                filename = "nba_data/" + sys.argv[2] + ".csv"
                np.savetxt(filename, np.array(output_table), delimiter=",")

        except:
            print "Error " + sys.argv[2]

    time.sleep(20)

filename = "nba_data/" + sys.argv[2] + "_" + sys.argv[3] + ".csv"
np.savetxt(filename, np.array(output_table), delimiter=",")
