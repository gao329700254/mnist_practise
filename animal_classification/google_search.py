import json
import requests
import random
import string
import shutil
import bs4
import os

def image_link(target_name):
    Res = requests.get("https://www.google.com/search?hl=jp&q=" + target_name + "&start=" + str(random.randrange(50) * 10) + "&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
    Html = Res.text
    Soup = bs4.BeautifulSoup(Html,'lxml')
    links = Soup.find_all("img")

    link = random.choice(links).get("src")
    while link.startswith('/'):
        link = random.choice(links).get("src")

    return link

def download_img(url, file_path):
    try:
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            return False

        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    except:
        return False

def generate_file_path(target_name):
    filename = randomstr()
    return os.path.join('data', target_name, filename + '.png')

def randomstr(length=10):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(length)])
