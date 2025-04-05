import requests
from bs4 import BeautifulSoup
import re

from urllib.parse import urlparse

# nutrisi
def scrape_nutrition_data(food_name, details=""):
    food_name = food_name.replace(" ", "-").lower()
    
    url = "https://www.fatsecret.co.id/kalori-gizi/umum/" + food_name + details
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", class_="generic spaced")
    
    # Label mapping berdasarkan prefix
    label_map = {
        "Kal": "Kalori",
        "Lemak": "Lemak",
        "Karb": "Karbohidrat",
        "Prot": "Protein"
    }

    result = {}

    if table:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            for col in cols:
                text = col.get_text(strip=True)
                for prefix, label in label_map.items():
                    if text.startswith(prefix):
                        match = re.search(r'\d+[.,]?\d*', text)
                        if match:
                            value = float(match.group().replace(",", "."))
                            result[label] = value
                        break  

    return result

# link porsi
def scrape_portion_links(food_name):
    food_name = food_name.replace(" ", "-").lower()
    
    url = "https://www.fatsecret.co.id/kalori-gizi/umum/" + food_name
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Label mapping berdasarkan prefix
    label_map = [
        "100 gram",
        "1 mangkok",
        "1 porsi",
        "1 tusuk",
    ]

    tables = soup.find_all("table", class_="generic")
    portion_links_dict = {}

    for table in tables:
        links = table.find_all("a", href=True)
        for link in links:
            text = link.get_text(strip=True)
            if text in label_map and text not in portion_links_dict:
                href = link["href"]
                # full_url = f"https://www.fatsecret.co.id{href}"
                parsed = urlparse(href)
                query = f"?{parsed.query}"
                portion_links_dict[text] = query


    portion_links = [{"text": key, "url": value} for key, value in portion_links_dict.items()]
    return portion_links

# porsi nutrisi
def scrape_portion_nutrition(food_name):
    food_name = food_name.replace(" ", "-").lower()
    
    portion_links = scrape_portion_links(food_name)
    
    portion_nutrition = []
    for portion in portion_links:
        portion_text = portion["text"]
        portion_url = portion["url"]
        
        nutrition_data = scrape_nutrition_data(food_name, portion_url)
        
        # Gabungkan data nutrisi dengan informasi porsi
        nutrition_data["porsi"] = portion_text
        portion_nutrition.append(nutrition_data)
        
    return portion_nutrition    