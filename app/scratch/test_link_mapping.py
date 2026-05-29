import sys
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

_APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_ROOT))

def main():
    html_path = _APP_ROOT / "scratch" / "main_stats.html"
    if not html_path.exists():
        print("Please run scratch/test_realgm_fetch.py first")
        return
        
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    
    # Let's find the stats table
    # The table probably has class "tablesaw" or similar, or just find all tables and check if they have "Player" column
    tables = soup.find_all("table")
    print(f"Found {len(tables)} tables")
    
    target_table = None
    for table in tables:
        # Check headers
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Player" in headers and "GP" in headers:
            target_table = table
            print("Found target table with headers:", headers[:10])
            break
            
    if not target_table:
        print("Did not find the stats table")
        return
        
    # For each row, let's find the Player column cell and its link
    rows = target_table.find("tbody").find_all("tr")
    print(f"Found {len(rows)} rows in tbody")
    
    mapped_players = []
    for i, row in enumerate(rows):
        cells = row.find_all("td")
        if not cells:
            continue
        # The player column is usually the first td (or index 1 if there's a rank column)
        # Let's search for an <a> link in any cell that has a summary page link
        player_link = None
        player_name = None
        for cell in cells:
            a_tag = cell.find("a", href=True)
            if a_tag and "/player/" in a_tag['href'] and "/Summary/" in a_tag['href']:
                player_link = a_tag['href']
                player_name = a_tag.get_text(strip=True)
                break
        
        if player_link:
            mapped_players.append((player_name, player_link))
            
    print(f"Successfully mapped {len(mapped_players)} player profile links.")
    print("First 10 mappings:")
    for name, link in mapped_players[:10]:
        print(f"  {name} -> {link}")

if __name__ == "__main__":
    main()
