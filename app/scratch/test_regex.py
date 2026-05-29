import re

def parse_height_weight(html: str) -> tuple[str, str]:
    # Extract just the imperial values: e.g. "6-9" and "210"
    # Height example: 6-9 (206cm)
    height_match = re.search(r'<strong>Height:</strong>\s*([0-9]+-[0-9]+)', html, re.IGNORECASE)
    # Weight example: 210 lbs (95kg)
    weight_match = re.search(r'<strong>Weight:</strong>\s*([0-9]+)', html, re.IGNORECASE)
    
    height = height_match.group(1).strip() if height_match else ""
    weight = weight_match.group(1).strip() if weight_match else ""
    return height, weight

# Let's test with mock strings
test_html_1 = "<p><strong>Height:</strong> 6-9 (206cm)</p><p><strong>Weight:</strong> 210 lbs (95kg)</p>"
test_html_2 = "<p><strong>Height:</strong> 6-11</p><p><strong>Weight:</strong> 245 lbs</p>"
test_html_3 = "<p><strong>Height:</strong> 7-0 (213cm)</p><p><strong>Weight:</strong> 250</p>"

for i, h in enumerate([test_html_1, test_html_2, test_html_3], 1):
    h_val, w_val = parse_height_weight(h)
    print(f"Test {i}: Height='{h_val}', Weight='{w_val}'")
