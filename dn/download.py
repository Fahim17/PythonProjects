import requests
from bs4 import BeautifulSoup
import re

url = 'http://tv-shows1.dhakamovie.com/sdg1/tv-shows1/2015/Two%20And%20A%20Half%20Men%201-7%20Seasons%20complete/Season%203/'

source_code = requests.get(url)
plain_text = source_code.text

soup = BeautifulSoup(plain_text)

name = soup.findAll('a', href=re.compile('.*avi*'))
print(len(name))

# for i in range(len(name)-1):
# 	sgl = str(name[i]).split('\"')		
# 	print(">>>>>>>>>>>>>>  ",sgl[1],"\n")

sgl = str(name[0]).split('\"')
print(">>>>>>>>>>>>>>\n",sgl[1])
host = "http://tv-shows1.dhakamovie.com/sdg1/tv-shows1/2015/Two%20And%20A%20Half%20Men%201-7%20Seasons%20complete/Season%203/"

dn = requests.get(host+sgl[1])

with open('tutorial.mp4', 'wb') as f:
    f.write(dn.content)