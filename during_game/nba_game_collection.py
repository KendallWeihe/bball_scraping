import pdb
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import datetime
import os
import sys

if len(sys.argv) != 2:
    print "Missing argument"
    sys.exit(0)

# scores_link = "http://www.espn.com/nba/scoreboard"
scores_link = "http://www.espn.com/nba/scoreboard/_/date/20161211"

r = requests.get(scores_link)
soup = BeautifulSoup(r.text, "html.parser")

scripts = soup.find_all('script')
pattern = re.compile('window.espn.scoreboardData')

for script in scripts:
    if(pattern.match(str(script.string))):
        team_indices = [i for i in range(len(str(script.string))) if str(script.string).startswith('shortDisplayName', i)]
        team_names = []
        for j in range(len(team_indices)):
            text = str(script.string[team_indices[j]+19:team_indices[j]+40])
            team = ""
            for k in range(len(text)):
                if text[k] == "\"":
                    break
                team = team + text[k]
            team_names.append(team)

        indices = [i for i in range(len(str(script.string))) if str(script.string).startswith('http://www.espn.com/nba/game?gameId=', i)]
        game_links = []
        for j in range(len(indices)):
            game_links.append(str(script.string[indices[j]:indices[j]+45]))

        time_indices = [i for i in range(len(str(script.string))) if str(script.string).startswith('shortDetail', i)]
        times = []
        # pdb.set_trace()
        for j in range(0,len(time_indices),2):
            text = str(script.string[time_indices[j]:time_indices[j]+100])
            if re.findall("\d+:\d+", text) != []:
                times.append(re.findall("\d+:\d+", text)[0])

adjusted_times = []
for i in range(len(times)):
    hour = ""
    for j in range(len(times[i])):
        if times[i][j] == ":":
            break
        hour = hour + times[i][j]
    hour = int(hour) + 12

    minutes = ""
    minutes_flag = False
    for j in range(len(times[i])):
        if minutes_flag:
            minutes = minutes + times[i][j]
        elif times[i][j] == ":":
            minutes_flag = True

    new_time = str(hour) + ":" + minutes
    adjusted_times.append(new_time)

adjusted_teams = []
for i in range(0,len(team_names),2):
    adjusted_teams.append([team_names[i], team_names[i+1]])

for i in range(len(game_links)):
    game_links[i] = game_links[i].replace("game?", "matchup?")

started_games = []
# pdb.set_trace()

print adjusted_times
print "\n"
print adjusted_teams
print "\n"
print game_links

while 1:

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time = re.findall("\d+:\d+",time)[0]
    # print time
    # pdb.set_trace()
    if time in adjusted_times:
        # print "time(s) found"
        for i in range(len(adjusted_times)):
            if time == adjusted_times[i] and i not in started_games:
                print "starting new game -- " + adjusted_teams[i][0] + " " + adjusted_teams[i][1]
                started_games.append(i)
                adjusted_teams[i][0] = adjusted_teams[i][0].replace(" ", "_")
                adjusted_teams[i][1] = adjusted_teams[i][1].replace(" ", "_")
                command = "python nba_stat_collection.py " + str(game_links[i]) + " " + str(adjusted_teams[i][0]) + "_" + str(adjusted_teams[i][1]) + " " + sys.argv[1] + " &"
                os.system(command)
