import requests
from bs4 import BeautifulSoup
import pdb
from selenium import webdriver
import unicodedata
import time
from difflib import SequenceMatcher

link = "https://sports.bovada.lv/basketball/college-basketball"

driver = webdriver.Chrome("/home/kendall/Development/bball_scraping/during_game/chromedriver")
driver.get(link)
for i in range(10):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)

soup = BeautifulSoup(driver.page_source, "html.parser")

games = soup.findAll('article')
# pdb.set_trace()
for game in games:
    try:
        time = game.findAll('time')[0].string
        if "Second Half" in time:
            team1 = unicodedata.normalize('NFKD', game.findAll('h3')[0].string).encode('ascii','ignore')
            team2 = unicodedata.normalize('NFKD', game.findAll('h3')[1].string).encode('ascii','ignore')

            if SequenceMatcher(None, team1, sys.argv[1]).ratio() > 0.5:
                spreadTeam1 = float(game.findAll('ul')[3].findAll('li')[0].findAll('span')[0].string)
                print team1
            elif SequenceMatcher(None, team1, sys.argv[2]).ratio() > 0.5:
                spreadTeam1 = float(game.findAll('ul')[3].findAll('li')[0].findAll('span')[1].string)
    except:
        pass
