import requests
from bs4 import BeautifulSoup

from food import food_list

url = "https://www.fatsecret.co.id/kalori-gizi/umum/bakso-daging-sapi"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

tables = soup.find_all("table", class_="generic")

for table in tables:
    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        data = [col.get_text(strip=True) for col in cols]
        print(data)


if __name__ == "__main__":
    food_list()