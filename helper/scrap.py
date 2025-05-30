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
    tabe_volume = soup.find("table", class_='generic')
    
    # Label mapping berdasarkan prefix
    label_map = {
        "Kal": "Kalori",
        "Lemak": "Lemak",
        "Karb": "Karbohidrat",
        "Prot": "Protein"
    }
    
    default_volume = 0
    result = {}

    if table:
        rows = table.   find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            for col in cols:
                text = col.get_text(strip=True)
                for prefix, label in label_map.items():
                    if text.startswith(prefix):
                        match = re.search(r'\d+[.,]?\d*', text)
                        if match:
                            value = str(match.group().replace(",", "."))
                            result[label] = value + " g" if label != "Kalori" else value + " kcal"
                        break  

    if tabe_volume:
        rows = tabe_volume.find("tr", class_="selected")
        
        if rows:
            cols = rows.find("td")
            if cols:
                text = cols.get_text(strip=True)
                default_volume = text
    
    return result, default_volume

# link porsi
def scrape_portion_links(food_name):
    """
    Scrape available portion options for a food item.
    
    Parameters:
        - food_name (str): Name of the food to search for
        
    Returns:
        - list: Dictionaries with portion text and URL query parameters
    """
    try:
        food_name = food_name.replace(" ", "-").lower()
        
        url = "https://www.fatsecret.co.id/kalori-gizi/umum/" + food_name
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Common portion types to look for
        label_map = [
            "100 gram",
            "1 mangkok", 
            "1 porsi",
            "1 tusuk",
            "1 gelas",
            "1 buah",
            "1 potong",
            "1 piring"
        ]

        tables = soup.find_all("table", class_="generic")
        portion_links_dict = {}

        for table in tables:
            links = table.find_all("a", href=True)
            for link in links:
                text = link.get_text(strip=True)
                if text in label_map and text not in portion_links_dict:
                    href = link["href"]
                    parsed = urlparse(href)
                    query = f"?{parsed.query}"
                    portion_links_dict[text] = query

        # Format the results for better structure
        portion_links = [
            {
                "text": key, 
                "url": value,
                "description": f"Porsi {key} untuk {food_name.replace('-', ' ')}"
            } 
            for key, value in portion_links_dict.items()
        ]
        
        return portion_links
    except Exception as e:
        print(f"Error scraping portion links: {e}")
        return []

# porsi nutrisi
def scrape_portion_nutrition(food_name):
    food_name = food_name.replace(" ", "-").lower()
    
    portion_links = scrape_portion_links(food_name)
    
    portion_nutrition = []
    for portion in portion_links:
        portion_text = portion["text"]
        portion_url = portion["url"]
        nutrition_data, volume = scrape_nutrition_data(food_name, portion_url)
        
        # Gabungkan data nutrisi dengan informasi porsi
        nutrition_data["porsi"] = portion_text
        nutrition_data["volume"] = volume
        portion_nutrition.append(nutrition_data)
        
    return portion_nutrition    