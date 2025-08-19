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

# search
# search
def scrape_search_list(query):
    """
    Search for food items based on a query string.
    
    Parameters:
        - query (str): The search query (food name)
        
    Returns:
        - list: A list of dictionaries containing food item details
    """
    try:
        query = query.replace(" ", "-").lower()
        url = f"https://www.fatsecret.co.id/kalori-gizi/search?q={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find search results
        results = []
        table = soup.find("table", class_="generic searchResult")

        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if cols:
                name = cols[0].find("a", href=True).get_text(strip=True)
                raw_desc = cols[0].find("div", class_="smallText greyText greyLink").get_text(" ", strip=True)
                

                def parse_description(text):
                    # normalize whitespace
                    t = re.sub(r'\s+', ' ', text).strip()

                    # remove trailing junk like ", lagi... Informasi Gizi - Mirip" or similar phrases
                    t = re.sub(r',?\s*lagi[^\n\r]*$', '', t, flags=re.IGNORECASE)
                    t = re.sub(r'Informasi\s*Gizi[^\n\r]*$', '', t, flags=re.IGNORECASE)
                    t = re.sub(r'-\s*Mirip[^\n\r]*$', '', t, flags=re.IGNORECASE)
                    # cleanup leftover separators/spaces
                    t = re.sub(r'[\s,;-]{2,}', ' ', t).strip(' ,;-.')

                    # normalize kkal -> kcal and ensure space before unit (e.g. '237kkal' -> '237 kcal', '13,49g' -> '13,49 g')
                    t = re.sub(r'(?i)kkal', 'kcal', t)
                    t = re.sub(r'(?i)(\d)(kcal)\b', r'\1 \2', t)
                    t = re.sub(r'(?i)(\d[0-9.,]*)(g|gr|gram)\b', r'\1 \2', t)

                    # try extract per 100 gram nutrition block
                    base = {}
                    m = re.search(
                        r'per\s*100\s*(?:gram|gr)\s*[-:]?\s*Kalori[:\s]*([\d.,]+\s*kcal).*?Lemak[:\s]*([\d.,]+\s*g).*?Karbohidrat[:\s]*([\d.,]+\s*g).*?Protein[:\s]*([\d.,]+\s*g)',
                        t, flags=re.IGNORECASE)
                    if m:
                        base = {
                            "Kalori": m.group(1).strip(),
                            "Lemak": m.group(2).strip(),
                            "Karbohidrat": m.group(3).strip(),
                            "Protein": m.group(4).strip()
                        }
                    else:
                        # fallback: try looser capture of numbers after labels
                        for k in ("Kalori","Lemak","Karbohidrat","Protein"):
                            mm = re.search(rf'{k}[:\s]*([\d.,]+\s*(?:kcal|g)?)', t, flags=re.IGNORECASE)
                            if mm:
                                base[k] = mm.group(1).strip()

                    # extract other portion entries after "Ukuran Lainnya" or "Ukuran Lainnya:"
                    portions = []
                    parts = re.split(r'Ukuran Lainnya[:\s]*', t, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        rest = parts[1]
                        # remove trailing junk from rest as well
                        rest = re.sub(r',?\s*lagi[^\n\r]*$', '', rest, flags=re.IGNORECASE)
                        rest = re.sub(r'Informasi\s*Gizi[^\n\r]*$', '', rest, flags=re.IGNORECASE)
                        # find patterns like "1 irisan tipis - 17kkal" or "1 mangkok, dimasak, potong dadu - 320kkal"
                        for m2 in re.finditer(r'([^,.-]+?)\s*[-–]\s*([\d.,]+\s*kcal)', rest, flags=re.IGNORECASE):
                            p_label = m2.group(1).strip().strip(",")
                            p_kcal = m2.group(2).strip()
                            portions.append({"porsi": p_label, "Kalori": p_kcal})

                    return base, portions, t

                _, _, cleaned = parse_description(raw_desc)
                description = cleaned

                results.append({"name": name, "description": description})

        return results
    except Exception as e:
        print(f"Error scraping search list: {e}")
        return []