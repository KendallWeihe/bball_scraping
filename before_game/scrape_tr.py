import numpy as np
import requests
from bs4 import BeautifulSoup
import pdb
import csv
import sys

if len(sys.argv) != 2:
    print "Missing argument"
    sys.exit(0)

team_hash_table = []
with open('team_hash_table.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

link = "https://www.teamrankings.com/ncb/schedules/?date=2017-01-21"
r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")

games = soup.find("table", class_="tr-table datatable scrollable")
game_links = games.findAll("a")
todays_stats = []
for game in game_links:
    try:
        game_str = str(game.string)
        team_1_rank = ""
        index = 0
        game_stats = []
        for i in range(len(game_str)):
            if game_str[i] == "#":
                for j in range(i+1,i+5):
                    if game_str[j] == " ":
                        index = j
                        break
                    else:
                        team_1_rank = team_1_rank + game_str[j]
                break

        team_1_name = ""
        for i in range(index+1, len(game_str)):
            if game_str[i] == " " and game_str[i+1] == " " and game_str[i+2] == "a" and game_str[i+3] == "t":
                index = i + 7
                break
            if game_str[i] == " " and game_str[i+1] == " " and game_str[i+2] == "v" and game_str[i+3] == "s":
                index = i + 8
                break
            team_1_name = team_1_name + game_str[i]
        team_1_name = team_1_name.replace(" ", "_").replace("\'", "")

        team_2_rank = ""
        for i in range(index, len(game_str)):
            if game_str[i] == " ":
                index = i + 1
                break
            team_2_rank = team_2_rank + game_str[i]

        team_2_name = ""
        for i in range(index, len(game_str)):
            team_2_name = team_2_name + game_str[i]
        team_2_name = team_2_name.replace(" ", "_").replace("\'", "")


        for i in range(len(team_hash_table)):
            if team_hash_table[i][0] == team_1_name:
                team_1_index = i
            if team_hash_table[i][0] == team_2_name:
                team_2_index = i

        game_stats.append(float(team_1_index))
        game_stats.append(float(team_2_index))
        game_stats.append(float(team_1_rank))
        game_stats.append(float(team_2_rank))

        game_link = "https://www.teamrankings.com" + str(game["href"])
        r = requests.get(game_link)
        inner_soup = BeautifulSoup(r.text, "html.parser")

        table = inner_soup.findAll("table", class_="tr-table matchup-table")[4]
        data = table.findAll("td")

        team_1_predicted_score = str(data[2].string)
        team_2_predicted_score = str(data[5].string)

        game_stats.append(float(team_1_predicted_score))
        game_stats.append(float(team_2_predicted_score))

        offensive_stats = inner_soup.findAll("table", class_="tr-table")[5].findAll("tr")
        del offensive_stats[0]
        offensive_stats_team_1 = []
        offensive_stats_team_2 = []
        for row in offensive_stats:
            spans = row.findAll("span")
            for span in spans:
                span.unwrap()
            rows = row.findAll("td")
            # offensive_stats_team_1.append(str(rows[1].string))
            # offensive_stats_team_2.append(str(rows[2].string))
            game_stats.append(float(rows[1].string.replace("%","")))
            game_stats.append(float(rows[2].string.replace("%","")))

        defensive_stats = inner_soup.findAll("table", class_="tr-table")[6].findAll("tr")
        del defensive_stats[0]
        defensive_stats_team_1 = []
        defensive_stats_team_2 = []
        for row in defensive_stats:
            spans = row.findAll("span")
            for span in spans:
                span.unwrap()
            rows = row.findAll("td")
            # defensive_stats_team_1.append(str(rows[1].string))
            # defensive_stats_team_2.append(str(rows[2].string))
            game_stats.append(float(rows[1].string.replace("%","")))
            game_stats.append(float(rows[2].string.replace("%","")))

        # pdb.set_trace()
        print game_stats
        print len(game_stats)
        todays_stats.append(game_stats)

    except:
        print "Error: " + team_1_name + " " + team_2_name

filename = "./stats/w_out_score/" + sys.argv[1] + ".csv"
np.savetxt(filename, np.array(todays_stats), delimiter=",", fmt="%s")
