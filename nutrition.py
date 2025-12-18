import os
import requests
from dotenv import load_dotenv

load_dotenv()
USDA_KEY = os.getenv("USDA_API_KEY")

# -----------------------------
# Search USDA + extract nutrients
# -----------------------------
def usda_search(term):
    if not USDA_KEY:
        return None

    # We removed the strict 'normalize' function because 
    # test_usda.py proved the raw string works better.
    # We just strip whitespace.
    term = term.strip()

    # Search endpoint
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "api_key": USDA_KEY, 
        "query": term, 
        "pageSize": 1
    }

    try:
        res = requests.get(search_url, params=params, timeout=5).json()
    except:
        return None

    foods = res.get("foods", [])
    if not foods:
        return None

    # DIRECTLY use the first match from the search (Like test_usda.py)
    # Do not make a second API call.
    food = foods[0]
    
    # Initialize dictionary
    nutrients = {"calories": None, "protein": None, "fat": None, "sugar": None}
    
    # Iterate through the nutrients found in the search result
    for n in food.get("foodNutrients", []):
        name = n.get("nutrientName", "").lower()
        value = n.get("value", 0.0)

        # Logic copied from test_usda.py
        if ("energy" in name or "kcal" in name) and nutrients["calories"] is None:
            nutrients["calories"] = value

        elif "protein" in name:
            nutrients["protein"] = value

        elif ("total lipid" in name or "fat" in name):
            nutrients["fat"] = value

        elif "sugar" in name:
            nutrients["sugar"] = value

    return nutrients


# -----------------------------
# Dish-first → fallback ingredient average
# -----------------------------
def nutrition_for_dish(dish_name: str, ingredients: list):
    # 1) Try direct dish match
    dish_nut = usda_search(dish_name)
    
    # Check if we actually got numbers back
    if dish_nut and (dish_nut["calories"] is not None or dish_nut["protein"] is not None):
        return dish_nut, f"Dish match: {dish_name}"

    # 2) Ingredient fallback
    values = {"calories": [], "protein": [], "fat": [], "sugar": []}

    for ing in ingredients:
        # minimal cleanup for ingredients
        clean_ing = ing.replace("-", " ").strip()
        nut = usda_search(clean_ing)
        
        if not nut:
            continue

        for k in values:
            if nut[k] is not None:
                values[k].append(nut[k])

    # If no data collected
    if not any(values[k] for k in values):
        return None, "No ingredient nutrition found"

    # Calculate averages
    avg = {}
    for k, arr in values.items():
        avg[k] = round(sum(arr)/len(arr), 2) if arr else 0

    return avg, f"Avg of {len(values['calories'])} ingredients"