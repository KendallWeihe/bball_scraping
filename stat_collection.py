import pdb
import numpy as np
import requests
from bs4 import BeautifulSoup
import re


scores_link = "http://www.espn.com/nba/scoreboard"

r = requests.get(scores_link)
soup = BeautifulSoup(r.text, "html.parser")

scripts = soup.find_all('script')
pattern = re.compile('window.espn.scoreboardData')

for script in scripts:
    if(pattern.match(str(script.string))):
       pdb.set_trace()
       indices = [i for i in range(len(str(script.string))) if str(script.string).startswith('http://www.espn.com/nba/game?gameId=', i)]
       game_links = []
       for j in range(len(indices)):
           game_links.append(str(script.string[indices[0]:indices[0]+45]))

pdb.set_trace()



link = "http://www.espn.com/nba/matchup?gameId=400899645"

r = requests.get(link)
soup = BeautifulSoup(r.text, "html.parser")
table = soup.find('table', class_="mod-data")
team_rows = table.findAll('tr')

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

pdb.set_trace()
print "here"
