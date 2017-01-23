import pdb
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import sys
import time
import os

# ARGV:
#     game link
#     team 1
#     team 2
#     date

link = sys.argv[1]
over = False
output_table = []
record_vegas_2nd_half = False
timeCount = 0
while not over:

    r = requests.get(link)
    soup = BeautifulSoup(r.text, "html.parser")

    game_time = soup.find('span', class_="game-time")
    game_time = str(game_time.string)

    # pdb.set_trace()
    if game_time == "Final/OT" or game_time == "Final":
        over = True

    elif game_time == "Halftime" and record_vegas_2nd_half == False:
        os.system("python record_vegas_2nd_half.py " + sys.argv[2] + " " + sys.argv[3] + " " + sys.argv[4] + " &")
        record_vegas_2nd_half = True
        try:
            spread_filename = "./half_time_spreads/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
            vegas_spread = np.genfromtxt(spread_filename, delimiter=",")
            stats_filename = "./ncaa_data/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
            stats = np.genfromtxt(stats_filename, delimiter=",")
            halftime_score = np.append(vegas_spread, stats[stats.shape[0],2]-stats[stats.shape[0],3])
            np.savetxt(spread_filename, halftime_score)
        except:
            print "Vegas spread not found: " + sys.argv[2] + " " + sys.argv[3]
        # TODO:
            # call program to:
                # open bovada
                # find game
                # find spread
                # run network & predict
                # send text message with results
        os.system("python CompareBovada.py " + sys.argv[2] + " " + sys.argv[3] + " " + sys.argv[4] + " &")

    elif game_time != "Halftime" and game_time != 'None':
        table = soup.find('table', class_="mod-data")
        # if timeCount % 5 == 0 and "2nd Half" in game_time:
        #     os.system("python CompareBovada.py " + sys.argv[2] + " " + sys.argv[3] + " " + sys.argv[4] + " &")

        if table != None:
            team_rows = table.findAll('tr')

            try:
                score_1 = str(soup.find('div', class_="score icon-font-after").string)
                score_2 = str(soup.find('div', class_="score icon-font-before").string)

                field_goal_percentages = re.findall("\d+\.\d+", str(team_rows[2]))
                three_point_percentages = re.findall("\d+\.\d+", str(team_rows[4]))
                free_throw_percentages = re.findall("\d+\.\d+", str(team_rows[6]))
                total_rebounds = re.findall("\d+", str(team_rows[7]))
                # offensive_rebounds = re.findall("\d+", str(team_rows[8]))
                # defensive_rebounds = re.findall("\d+", str(team_rows[9]))
                assists = re.findall("\d+", str(team_rows[11]))
                steals = re.findall("\d+", str(team_rows[12]))
                blocks = re.findall("\d+", str(team_rows[13]))
                turnovers = re.findall("\d+", str(team_rows[14]))
                personal_fouls = re.findall("\d+", str(team_rows[15]))

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

                filename = "./ncaa_data/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
                np.savetxt(filename, np.array(output_table), delimiter=",")

            except:
                print "There was an error -- " + sys.argv[2]

    time.sleep(20)
    timeCount = timeCount + 1

filename = "ncaa_data/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
np.savetxt(filename, np.array(output_table), delimiter=",")

try:
    spread_filename = "./half_time_spreads/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
    vegas_spread = np.genfromtxt(spread_filename, delimiter=",")
    stats_filename = "./ncaa_data/" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
    stats = np.genfromtxt(stats_filename, delimiter=",")
    final_spread = np.append(vegas_spread, stats[stats.shape[0],2]-stats[stats.shape[0],3])
    np.savetxt(spread_filename, final_spread)
except:
    print "Vegas spread not found: " + sys.argv[2] + " " + sys.argv[3]
