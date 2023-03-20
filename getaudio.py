import requests
# import urllib.request
import re
import os
from bs4 import BeautifulSoup
requests.packages.urllib3.disable_warnings()

#Download mp3 file from a list of urls of https://dsprojects.lib.cuhk.edu.hk/en/projects/20th-cantonese-poetry-chanting/
#to folder 'raw_audio'    

path = './raw_audio/'

with open('urls.txt', 'r') as f:
    urls = f.read().split('\n')

for url in urls:
    
    #Create folder if not exist
    foldername = url[url.rfind("/")+1:]
    if not os.path.exists(path+foldername):
        os.makedirs(path+foldername)

    r = requests.get(url, verify=False)
    soup = BeautifulSoup(r.content, 'html.parser')

    for a in soup.find_all('audio', src=re.compile(r'https.*\.mp3')):
        filename = a['src'][a['src'].rfind("/")+1:]
        doc = requests.get(a['src'], verify=False)
        with open(path+foldername+'/'+filename, 'wb') as f:
            f.write(doc.content)
