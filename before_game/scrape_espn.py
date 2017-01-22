import numpy as np
import requests
from bs4 import BeautifulSoup
import pdb
from selenium import webdriver
import time
import csv
import sys

# ARGV:
    # w_out_score filename
    # date

team_hash_table = []
with open("./team_hash_table.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        team_hash_table.append(row)

tr_data = np.genfromtxt(sys.argv[1], delimiter=",")

scores_link = "http://www.espn.com/mens-college-basketball/scoreboard/_/date/20" + sys.argv[2]
r = requests.get(scores_link)

driver = webdriver.Chrome("/home/kendall/Development/bball_scraping/during_game/chromedriver")
driver.get(scores_link)

# pdb.set_trace()
time.sleep(1)
driver.find_element_by_css_selector("#scoreboard-page header .dropdown-type-group button").click()
time.sleep(1)
driver.find_element_by_link_text('NCAA Division I').click()
time.sleep(1)

soup = BeautifulSoup(driver.page_source)
games = soup.find("div", {"id": "events"})

final_stats = []
count = 0
for game in games:
    try:
        team_1 = str(game.find_all("span", {"class": "sb-team-short"})[0].string.encode('utf-8')).replace(" ", "_").replace("\'","")
        team_2 = str(game.find_all("span", {"class": "sb-team-short"})[1].string.encode('utf-8')).replace(" ", "_").replace("\'","")
        team_1_score = float(game.findAll("td", class_="total")[0].string)
        team_2_score = float(game.findAll("td", class_="total")[1].string)

        for i in range(len(team_hash_table)):
            if team_1 == team_hash_table[i][1]:
                team_1_index = i
            if team_2 == team_hash_table[i][1]:
                team_2_index = i

        for i in range(tr_data.shape[0]):
            if tr_data[i,0] == team_1_index:
                temp_list = tr_data[i,:].tolist()
                temp_list.append(team_1_score)
                temp_list.append(team_2_score)
                final_stats.append(temp_list)

        cout = count + 1
    except:
        print "count = " + str(count)
        # pdb.set_trace()
        print "failed"


print "num games = " + str(len(final_stats))
final_stats = np.array(final_stats)

#TODO
    # check if file already exists
filename = "./stats/w_score/" + sys.argv[2] + ".csv"
np.savetxt(filename, final_stats, delimiter=",")

driver.quit()
