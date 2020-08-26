# usage example
# python3 img_scraping.py -t cat -n 100 -f true

import argparse
import os
import json
from google_search import *
from img_resize import *

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument("-t", "--target",  help="target name", type=str, required=True)
parser.add_argument("-n", "--number", help="number of images", type=int, required=True)
parser.add_argument("-f", '--forc', help="download overwrite existing file", type=bool, default=False)

args = parser.parse_args()

target_name = args.target

# HACK: 再利用の際に、このファイルパスを調整する
# 基本的にワーキング・ディレクトリに入ってから、このファイルを実行
os.makedirs(os.path.join('data', target_name), exist_ok=True)

downloaded_count = 0

while downloaded_count < int(args.number):
    link = image_link(target_name)
    file_path = generate_file_path(target_name)

    if download_img(link, file_path) == False:
        print("drop " + link)
        continue

    resize_img(file_path)
    downloaded_count += 1
    print(' '.join(['Downloaded', target_name, str(downloaded_count)]))
print("OK")
