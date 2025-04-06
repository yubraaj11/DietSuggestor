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

