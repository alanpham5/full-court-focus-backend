import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parents[1]
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from scripts.build_prospects_dataset import fetch_html

# Let's fetch AJ Dybantsa's summary profile
url = "https://basketball.realgm.com/player/AJ-Dybantsa/Summary/186435"
html = fetch_html(url)
for line in html.splitlines():
    if "position" in line.lower() or "height" in line.lower() or "weight" in line.lower():
        print(line.strip())
