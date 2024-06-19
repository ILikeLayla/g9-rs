import pyhttpx
import urllib.parse
from bs4 import BeautifulSoup
import time

CHAR = "1234567890."

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
}

def get_price(product_name):
    encoded_string = urllib.parse.quote(product_name)
    session = pyhttpx.HttpSession()
    res = session.get(url=f'https://www.tbkong.com/search?keyword={encoded_string}',headers=HEADERS).text

    soup = BeautifulSoup(res, 'html.parser')

    price = []

    for each_text in soup.find_all(attrs={'class': 'main_price'}):
        raw_price = each_text.get_text()
        real_price = ""
        for each_char in raw_price:
            if each_char in CHAR:
                real_price += each_char
        if real_price != "":
            price.append(float(real_price))
    
    return price

prices = []

with open(r"C:\code\g9-rs\data\name_zh.txt", "r", encoding="utf-8") as f1:
    for line in f1.readlines():
        line = line.strip("\n") + " 环保"
        price = get_price(line)
        average_price = sum(price) / len(price) if price else 0.00
        prices.append(average_price)
        print(f"average price of #{len(prices)} {line}: {average_price}")
        time.sleep(0.5)

with open(r"C:\code\g9-rs\data\prices_recycle.txt", "w", encoding="utf-8") as f:
    for price in prices:
        f.write(str(price) + "\n")