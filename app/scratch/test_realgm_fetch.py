import sys
from pathlib import Path

# Add app and app/scripts directories to path
_APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP_ROOT))
sys.path.insert(0, str(_APP_ROOT / "scripts"))

from scripts.build_prospects_dataset import fetch_html, DEFAULT_SOURCE_URL

def test_fetch():
    print("Fetching main stats page...")
    html = fetch_html(DEFAULT_SOURCE_URL)
    print("Length of main stats HTML:", len(html))
    
    # Write to scratch to inspect a bit
    scratch_dir = _APP_ROOT / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    (scratch_dir / "main_stats.html").write_text(html, encoding="utf-8")
    
    # Try finding some player links. Player links in RealGM normally look like:
    # <a href="/player/AJ-Dybantsa/Summary/186435">A.J. Dybantsa</a>
    # Let's search for this pattern.
    import re
    player_links = re.findall(r'href="(/player/[^"]+/Summary/\d+)"', html)
    print(f"Found {len(player_links)} player links.")
    if player_links:
        print("First 5 player links:", player_links[:5])
        
        # Now fetch the first player's profile page
        player_url = "https://basketball.realgm.com" + player_links[0]
        print(f"Fetching profile page: {player_url}")
        profile_html = fetch_html(player_url)
        (scratch_dir / "player_profile.html").write_text(profile_html, encoding="utf-8")
        print("Saved profile HTML to scratch/player_profile.html")
        
        # Let's extract height and weight lines
        height_match = re.search(r'Height:.*?</p>', profile_html, re.IGNORECASE)
        weight_match = re.search(r'Weight:.*?</p>', profile_html, re.IGNORECASE)
        if height_match:
            print("Height match:", height_match.group(0))
        else:
            print("Height not found using basic pattern")
        if weight_match:
            print("Weight match:", weight_match.group(0))
        else:
            print("Weight not found using basic pattern")
    else:
        print("No player links found.")

if __name__ == "__main__":
    test_fetch()
