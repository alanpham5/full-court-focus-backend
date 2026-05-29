import sys
import json
from pathlib import Path

# Add app to path
app_dir = Path(__file__).resolve().parents[1]
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from config import PROSPECTS_JSON_PATH
from routers.draft import list_prospects, get_prospect

# Mocking FastAPI request.app.state
class DummyState:
    def __init__(self, prospects):
        self.prospects = prospects
        self.prospects_by_id = {p["prospect_id"]: p for p in prospects}

class DummyApp:
    def __init__(self, prospects):
        self.state = DummyState(prospects)

class DummyRequest:
    def __init__(self, app):
        self.app = app

# Load actual prospects data to populate the mock app state
with PROSPECTS_JSON_PATH.open(encoding="utf-8") as f:
    prospects_list = json.load(f)

app_mock = DummyApp(prospects_list)
request_mock = DummyRequest(app_mock)

# Test list_prospects route
print("--- Calling list_prospects route directly ---")
results = list_prospects(request_mock)
print("Number of prospects returned:", len(results))
if len(results) > 0:
    first = results[0]
    # first is a ProspectListItem Pydantic model
    print("First prospect in list response:")
    print("  Name:", first.player_name)
    print("  Height:", first.height)
    print("  Weight:", first.weight)
    print("  Role:", first.role)
    print("  Raw Stats keys:", list(first.raw_stats.keys()))
    
    assert first.height == "6-9"
    assert first.weight == "210"
    assert first.role == "Designated Scorer"
    assert not hasattr(first, "profile_link")
    print("✓ height/weight/role checked successfully in list response")

# Test get_prospect route
print("\n--- Calling get_prospect route directly for 'a-j-dybantsa' ---")
detail = get_prospect("a-j-dybantsa", request_mock)
print("First prospect detail response:")
print("  Name:", detail.get("player_name"))
print("  Height:", detail.get("height"))
print("  Weight:", detail.get("weight"))
print("  Role:", detail.get("role"))
print("  Keys:", list(detail.keys()))

assert detail["height"] == "6-9"
assert detail["weight"] == "210"
assert detail["role"] == "Designated Scorer"
# Verify profile_link is not present in detail or in output
assert "profile_link" not in detail
print("✓ height/weight/role checked successfully in detail response")
