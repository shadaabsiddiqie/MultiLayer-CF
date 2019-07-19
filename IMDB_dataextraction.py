from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import csv
import numpy as np

urlInfo = ['movie_id', 'movie_url']
dfurl = pd.read_csv('movie_url_changed.csv',names=urlInfo)
urlDict = {}     #movie_id starts from 1 to 1683
for row in dfurl.itertuples():
    urlDict[row[1]] = row[2]

directorDict = {}           # soup.table.strings
                            # soup.select('table.simpleCreditsTable td.name a')
                            # css_soup.p['class']
                            # soup.find(id="simpleTable simpleCreditsTable")
                            # K.a['href']
                            # soup.select('table.simpleCreditsTable')[0].find_all('a')[3]['href']#all directors
castDict = {}
i = 0

MD = 'MovieDirector.csv'
MC = 'MovieCast.csv'
MC15 = 'MovieCast15.csv'
MC10 = 'MovieCast10.csv'
for key in urlDict.keys():
    url = urlDict[key]+'fullcredits/'
    print(key,url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    # with open(MD, 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     MovieDirectorArr = np.zeros((1245))
    #     for dire in soup.select('table.simpleCreditsTable')[0].find_all('a'):    
    #         st = dire['href']
    #         k = st.split('/')
    #         if k[2] not in directorDict.keys():
    #             directorDict[k[2]]=i
    #             i = i + 1
    #         MovieDirectorArr[directorDict[k[2]]]=1
    #     writer.writerow(MovieDirectorArr.astype(int))
    # csvFile.close()

    # -----------------------cast_list---------------------
    # max_cast = 10
    with open(MC, 'a') as csvFile:
        writer = csv.writer(csvFile)
        MovieCastArr = np.zeros((54853))
        for castmembers in soup.select('td.primary_photo a'):
            st = castmembers['href']                # print(castmembers.img['alt']) all names
            k = st.split('/')
            if k[2] not in castDict.keys():
                castDict[k[2]]=i
                i = i + 1
                # max_cast = max_cast - 1
            MovieCastArr[castDict[k[2]]]=1
            # if max_cast == 0:
            #     break
        writer.writerow(MovieCastArr.astype(int))
    csvFile.close()

print("NO OF Directors:",i)

#no of Directors :1245 done
#no of total CastMembers :54853
#no of top 10 CastMenbers :16174 
#no of top 15 CastMenbers :23373 doing
print(directorDict)