import os 
import requests
from dotenv import load_dotenv

load_dotenv()

API_NINJA = os.getenv("API_NINJA")

def vegetarian_filter(meal_options):
    """
    Filter or replace non-vegetarian meals with vegetarian alternatives.
    
    Args:
        meal_options: List of meal names
        
    Returns:
        List of vegetarian meal options
    """
    veg_alternatives = {
        'chicken': 'tofu',
        'beef': 'tempeh',
        'pork': 'jackfruit',
        'lamb': 'mushroom',
        'fish': 'tempeh',
        'salmon': 'eggplant',
        'tuna': 'chickpea',
        'shrimp': 'mushroom',
        'prawn': 'mushroom',
        'mutton': 'seitan',
        'turkey': 'seitan',
        'bacon': 'tempeh',
        'ham': 'seitan',
        'sausage': 'tofu',
        'meat': 'protein',
        'seafood': 'vegetables',
        'steak': 'cauliflower',
        'veal': 'jackfruit',
        'duck': 'seitan',
        'ribs': 'tempeh',
        'brisket': 'jackfruit',
        'crab': 'artichoke',
        'lobster': 'hearts-of-palm',
        'oyster': 'mushroom',
        'clam': 'mushroom',
        'mussel': 'mushroom',
        'squid': 'mushroom',
        'octopus': 'mushroom',
        'anchovies': 'olives'
    }
    
    vegetarian_options = []
    
    for meal in meal_options:
        original_meal = meal
        meal_lower = meal.lower()
        
        for non_veg, veg_alt in veg_alternatives.items():
            if non_veg in meal_lower or f"{non_veg}s" in meal_lower:
                meal_lower = meal_lower.replace(non_veg, veg_alt)
                meal_lower = meal_lower.replace(f"{non_veg}s", f"{veg_alt}s")
        
        if meal_lower != original_meal.lower():
            meal = ' '.join(word.capitalize() for word in meal_lower.split())
        else:
            meal = original_meal
            
        vegetarian_options.append(meal)
    
    return vegetarian_options



def fetch_nutrition_data(query):
    """Fetches nutrition data from the API."""
    api_url = f'https://api.api-ninjas.com/v1/nutrition?query={query}'
    response = requests.get(api_url, headers={'X-Api-Key': API_NINJA})

    if response.status_code == requests.codes.ok:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

def calculate_macronutrients(data):
    """Calculates total macronutrient values from API response."""
    macro_nutrients = {
        "Calories": 0,
        "Protein": 0,
        "Carbohydrates": 0,
        "Fat": 0,
        "Fiber": 0
    }

    if not data:
        return macro_nutrients

    for item in data:
        macro_nutrients["Calories"] += item.get("calories", 0) if isinstance(item.get("calories"), (int, float)) else 0
        macro_nutrients["Protein"] += item.get("protein_g", 0) if isinstance(item.get("protein_g"), (int, float)) else 0
        macro_nutrients["Carbohydrates"] += item.get("carbohydrates_total_g", 0) if isinstance(item.get("carbohydrates_total_g"), (int, float)) else 0
        macro_nutrients["Fat"] += item.get("fat_total_g", 0) if isinstance(item.get("fat_total_g"), (int, float)) else 0
        macro_nutrients["Fiber"] += item.get("fiber_g", 0) if isinstance(item.get("fiber_g"), (int, float)) else 0

    return macro_nutrients