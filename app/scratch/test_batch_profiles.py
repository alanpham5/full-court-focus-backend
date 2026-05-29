import sys
from pathlib import Path
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

_APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_ROOT))
sys.path.insert(0, str(_APP_ROOT / "scripts"))

from scripts.build_prospects_dataset import fetch_html

def parse_height_weight(html: str) -> tuple[str, str]:
    # Using regex to extract Height and Weight
    height_match = re.search(r'<strong>Height:</strong>\s*([^<]+)', html, re.IGNORECASE)
    weight_match = re.search(r'<strong>Weight:</strong>\s*([^<]+)', html, re.IGNORECASE)
    
    height = height_match.group(1).strip() if height_match else ""
    weight = weight_match.group(1).strip() if weight_match else ""
    return height, weight

def fetch_player_data(player_name, path):
    url = f"https://basketball.realgm.com{path}"
    try:
        html = fetch_html(url)
        height, weight = parse_height_weight(html)
        return player_name, height, weight, None
    except Exception as e:
        return player_name, "", "", str(e)

def main():
    test_players = [
        ("A.J. Dybantsa", "/player/AJ-Dybantsa/Summary/186435"),
        ("Darius Acuff, Jr.", "/player/Darius-Acuff-Jr/Summary/216187"),
        ("Ebuka Okorie", "/player/Ebuka-Okorie/Summary/193635"),
        ("Nick Martinelli", "/player/Nick-Martinelli/Summary/200150"),
        ("Cameron Boozer", "/player/Cameron-Boozer/Summary/184287"),
    ]
    
    print("Fetching player profiles in parallel...")
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_player_data, name, path): (name, path)
            for name, path in test_players
        }
        for future in as_completed(futures):
            name, height, weight, error = future.result()
            if error:
                print(f"Error fetching {name}: {error}")
            else:
                print(f"Parsed {name}: Height = '{height}', Weight = '{weight}'")

if __name__ == "__main__":
    main()
