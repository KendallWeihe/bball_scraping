import pdb
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import datetime
import os
from selenium import webdriver
import time


# scores_link = "http://www.espn.com/mens-college-basketball/scoreboard"
scores_link = "http://www.espn.com/mens-college-basketball/scoreboard/_/date/20161204"
r = requests.get(scores_link)

driver = webdriver.Chrome("/home/kendall/Documents/Development/bball_scraping/chromedriver")
driver.get(scores_link)

time.sleep(1)
driver.find_element_by_css_selector("#scoreboard-page header .dropdown-type-group button").click()
time.sleep(1)
driver.find_element_by_link_text('NCAA Division I').click()
time.sleep(1)

soup = BeautifulSoup(driver.page_source)
games = soup.find("div", {"id": "events"})

count = 0
times = []
teams = []
game_links = []
# pdb.set_trace()
for game in games:
    try:
        if str(re.findall("\d+:\d+ AM ET", str(game.find_all("th", {"class": "date-time"})))) == "[]":
            times.append(str(re.findall("\d+:\d+ PM ET", str(game.find_all("th", {"class": "date-time"})))[0]))
        else:
            times.append(str(re.findall("\d+:\d+ AM ET", str(game.find_all("th", {"class": "date-time"})))[0]))
        team_1 = str(game.find_all("span", {"class": "sb-team-short"})[0].string.encode('utf-8')).replace(" ", "_").replace("\'","")
        team_2 = str(game.find_all("span", {"class": "sb-team-short"})[1].string.encode('utf-8')).replace(" ", "_").replace("\'","")
        teams.append([team_1, team_2])
        game_links.append("http://www.espn.com/mens-college-basketball/matchup?gameId=" + str(game.get("id")))
        cout = count + 1
    except:
        print "count = " + str(count)
        #pdb.set_trace()
        print "failed"

# pdb.set_trace()
adjusted_times = []
for i in range(len(times)):
    hour = ""
    for j in range(len(times[i])):
        if times[i][j] == ":":
            break
        hour = hour + times[i][j]

    if re.findall("PM", times[i]) != []:
        if int(hour) != 12:
            hour = int(hour) + 12

    minutes = ""
    minutes_flag = False
    for j in range(len(times[i])):
        if times[i][j] == " ":
            break
        if minutes_flag:
            minutes = minutes + times[i][j]
        elif times[i][j] == ":":
            minutes_flag = True

    new_time = str(hour) + ":" + minutes
    adjusted_times.append(new_time)

print adjusted_times
print "\n"
print len(adjusted_times)
print "\n"
print teams
print "\n"
print len(teams)
print "\n"
print game_links
print "\n"
print len(game_links)

started_games = []
# pdb.set_trace()
while 1:

    time = datetime.datetime.now().strftime("%Y-%m-%d %-H:%M:%S")
    time = re.findall("\d+:\d+",time)[0]
    # print time
    # pdb.set_trace()
    if time in adjusted_times:
        # print "time(s) found"
        for i in range(len(adjusted_times)):
            if time == adjusted_times[i] and i not in started_games:
                print "starting new game -- " + teams[i][0] + " " + teams[i][1]
                started_games.append(i)
                teams[i][0].replace(" ", "_")
                teams[i][1].replace(" ", "_")
                command = "python ncaa_stat_collection.py " + str(game_links[i]) + " " + str(teams[i][0]) + "_" + str(teams[i][1]) + " &"
                os.system(command)
