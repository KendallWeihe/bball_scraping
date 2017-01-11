# TODO
# PROGRAM 1: CompareVegasToActual.py
    # for vegas_file in ./half_time_spreads:
    #     generate all combination of team names
    #         NOTE: the trailing date shall be spliced off
    #     for espn_link in dates:
    #         if game found:
    #             spread at half = team_1 - team_2
    #             final spread =
    #             difference =
    #             break espn_link loop
    #     multiply difference by -1 (to match vegas style)
    #     compare to vegas value

import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import time
import pdb
from selenium import webdriver
import math

# GLOBALS:
vegasFiles = os.listdir("./half_time_spreads/")
espnLinks = []
for i in range(20161228,20161232):
    espnLinks.append("http://www.espn.com/mens-college-basketball/scoreboard/_/group/50/date/" + str(i))
for i in range(20170101,20170110):
    espnLinks.append("http://www.espn.com/mens-college-basketball/scoreboard/_/group/50/date/" + str(i))


def generateTeamNames(n, spreadFile):
    teams1 = []
    teams2 = []
    splitIndices = []
    for i in range(n):
        team1 = ""
        team2 = ""
        secondTeam = False
        for j in range(len(spreadFile)-11):
            if not secondTeam:
                if spreadFile[j] == "_":
                    if j in splitIndices:
                        team1 = team1 + " "
                    else:
                        splitIndices.append(j)
                        secondTeam = True
                else:
                    team1 = team1 + spreadFile[j]
            else:
                if spreadFile[j] == "_":
                    team2 = team2 + " "
                else:
                    team2 = team2 + spreadFile[j]
        teams1.append(team1)
        teams2.append(team2)
    return teams1, teams2

def find2H(link, teams1, teams2):
    r = requests.get(link)

    driver = webdriver.Chrome("/home/kendall/Development/bball_scraping/during_game/chromedriver")
    driver.get(link)

    time.sleep(.3)
    driver.find_element_by_css_selector("#scoreboard-page header .dropdown-type-group button").click()
    time.sleep(.3)
    driver.find_element_by_link_text('NCAA Division I').click()
    time.sleep(.3)

    soup = BeautifulSoup(driver.page_source)
    games = soup.find("div", {"id": "events"})

    try:
        for game in games:
            team1 = str(game.find_all("span", {"class": "sb-team-short"})[0].string.encode('utf-8'))
            team2 = str(game.find_all("span", {"class": "sb-team-short"})[1].string.encode('utf-8'))

            if team1 in teams1 and team2 in teams2:
                team1HalftimeScore = float(game.find_all("td")[1].string.encode("utf-8"))
                team2HalftimeScore = float(game.find_all("td")[5].string.encode("utf-8"))
                halftimeSpread = team1HalftimeScore - team2HalftimeScore

                team1FinalScore = float(game.find_all("td")[3].string.encode("utf-8"))
                team2FinalScore = float(game.find_all("td")[7].string.encode("utf-8"))
                finalSpread = team1FinalScore - team2FinalScore

                secondHalfSpread = finalSpread - halftimeSpread
                driver.quit()
                return secondHalfSpread

        driver.quit()
        return -1000
    except:
        driver.quit()
        return -1000

absoluteDifferences = []
for spreadFile in vegasFiles:
    vegasSpread2H = float(np.genfromtxt("./half_time_spreads/" + spreadFile, delimiter=","))

    n = 0
    for i in range(len(spreadFile)-11):
        if spreadFile[i] == "_":
            n = n + 1

    teams1, teams2 = generateTeamNames(n, spreadFile)

    for link in espnLinks:
        actualSecondHalfSpread = find2H(link, teams1, teams2)
        if actualSecondHalfSpread > -1000:
            break

    actualSecondHalfSpread = actualSecondHalfSpread * -1
    absoluteDifferences.append(math.fabs(vegasSpread2H - actualSecondHalfSpread))
    print absoluteDifferences
    print "\n"
