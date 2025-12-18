import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

KEY = os.getenv("USDA_API_KEY")
print("Key loaded:", KEY)

url = "https://api.nal.usda.gov/fdc/v1/foods/search"
params = {
    "api_key": KEY,
    "query": "Medu Vada With Onion (Savoury Donuts) Recipe",
    "pageSize": 1
}

response = requests.get(url, params=params)
print("Status:", response.status_code)

data = response.json()
print("\n--- RAW USDA RESPONSE (first 500 chars) ---")
print(response.text[:500], "...\n")

foods = data.get("foods", [])
if not foods:
    print("❌ No foods found!")
    exit()

food = foods[0]   # first match

nutrients = food.get("foodNutrients", [])

# -------------------------------
# Extract required nutrients
# -------------------------------

result = {
    "calories": None,
    "protein": None,
    "fat": None,
    "sugar": None
}

for n in nutrients:
    name = n.get("nutrientName", "").lower()
    value = n.get("value")

    # Calories
    if ("energy" in name or "kcal" in name) and result["calories"] is None:
        result["calories"] = value

    # Protein
    elif "protein" in name:
        result["protein"] = value

    # Fat
    elif "total lipid" in name or "fat" in name:
        result["fat"] = value

    # Sugar
    elif "sugar" in name:
        result["sugar"] = value

print("\n--- Extracted Nutrients ---")
print(json.dumps(result, indent=4))
