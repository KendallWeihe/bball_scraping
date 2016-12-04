import numpy as np
import requests
from bs4 import BeautifulSoup
import pdb

# TEAMRANKINGS ____________________________________

link = "https://www.teamrankings.com/ncb/"
r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")

table = soup.find("table", id="league-overview-table")
team_rows = table.findAll("tr")
del team_rows[0]

tr_teams = []
for row in team_rows:
    team = str(row.findAll("a")[1].string).replace(" ", "_")
    tr_teams.append(team)
    # print team + ","



from selenium import webdriver
import time


scores_links = []
for i in range(17,28):
    scores_links.append("http://www.espn.com/mens-college-basketball/scoreboard/_/group/50/date/201611" + str(i))

espn_teams = []
for link in scores_links:

    driver = webdriver.Chrome("/home/kendall/Documents/Development/bball_scraping/chromedriver")
    driver.get(link)

    time.sleep(0.5)
    driver.find_element_by_css_selector("#scoreboard-page header .dropdown-type-group button").click()
    time.sleep(0.5)
    driver.find_element_by_link_text('NCAA Division I').click()
    time.sleep(0.5)

    soup = BeautifulSoup(driver.page_source)
    games = soup.find("div", {"id": "events"})

    for game in games:
        team_1 = str(game.find_all("span", {"class": "sb-team-short"})[0].string.encode('utf-8')).replace(" ", "_").replace("\'", "")
        team_2 = str(game.find_all("span", {"class": "sb-team-short"})[1].string.encode('utf-8')).replace(" ", "_").replace("\'", "")

        if team_1 not in espn_teams: espn_teams.append(team_1)
        if team_2 not in espn_teams: espn_teams.append(team_2)

pdb.set_trace()

# ESPN ______________________________________________

# link = "http://www.espn.com/mens-college-basketball/teams"
# r = requests.get(link)
# soup = BeautifulSoup(r.text, "html.parser")
#
# table = soup.find("div", class_="span-4")
# team_headers = table.findAll("h5")
#
# espn_teams = []
# for row in team_headers:
#     team = str(row.string).replace(" ", "_")
#     espn_teams.append(team)
#
equivalent_teams = []
unknown_indices = []
for i in range(len(tr_teams)):
    found = False
    for j in range(len(espn_teams)):
        if tr_teams[i] == espn_teams[j]:
            equivalent_teams.append([tr_teams[i], espn_teams[j]])
            found = True
    if found == False:
        unknown_indices.append(i)

for team in equivalent_teams:
    print team[0] + "," + team[1]

print "\n\n\n\n\n"

for index in unknown_indices:
    print tr_teams[index] + ","

print "\n\n\n\n\n"

for team in espn_teams:
    if team not in equivalent_teams:
        print team

pdb.set_trace()
print "here"
