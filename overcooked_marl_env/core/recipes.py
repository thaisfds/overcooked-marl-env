from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict
from .items import ItemType

@dataclass
class Recipe:
    name: str
    ingredients: List[ItemType]
    reward: float = 10.0
    time_limit: int = 100

    @property
    def type(self):
        """Recipe type is its name in uppercase for consistency"""
        return self.name.upper()

# Available recipe definitions in the environment
RECIPE_DEFINITIONS: Dict[str, Recipe] = {
    "SALAD": Recipe(
        name="SALAD",
        ingredients=[ItemType.TOMATO, ItemType.LETTUCE],
        reward=10.0
    ),
    "BURGUER": Recipe(
        name="BURGUER", 
        ingredients=[ItemType.LETTUCE, ItemType.MEAT],
        reward=15.0
    )
    # To add new recipes, simply add them here following the same format
}

class RecipeManager:
    def __init__(self, item_manager):
        self.item_manager = item_manager
        self.recipes = {}
        self.FINAL_TO_BIT = {ItemType.DISH: 1}
        self.ALL_FINAL_CODES = []

    def add_recipe(self, recipe: Recipe):
        """Adds a new recipe and updates the bit-wise mapping"""
        self.recipes[recipe.type] = recipe
        bit_value = 2
        for ingredient in recipe.ingredients:
            if ingredient not in self.FINAL_TO_BIT:
                self.FINAL_TO_BIT[ingredient] = bit_value
                bit_value *= 2
                self.ALL_FINAL_CODES.append(ingredient)

    def get_recipe_code(self, recipe_name: str) -> int:
        """Calculates the expected final code for a recipe."""
        recipe = self.recipes.get(recipe_name.upper())
        if not recipe:
            return 0
            
        code = self.FINAL_TO_BIT[ItemType.DISH]  # Start with dish (1)
        for ingredient in recipe.ingredients:
            code += self.FINAL_TO_BIT[ingredient]
        return code

    def is_final_ingredient(self, item_code) -> bool:
        """Checks if the item is a valid final ingredient."""
        return item_code in self.ALL_FINAL_CODES

    def get_combination_value(self, base_code: int, ingredient_code: int) -> int:
        """Combines a base code with a new ingredient."""
        bit_value = self.FINAL_TO_BIT.get(ingredient_code, 0)
        return base_code + bit_value


# Helper functions for recipe management
def get_available_recipes() -> List[str]:
    """Returns a list with the names of all available recipes"""
    return list(RECIPE_DEFINITIONS.keys())

def get_default_recipe() -> str:
    """Returns the default recipe name"""
    return list(RECIPE_DEFINITIONS.keys())[0] if RECIPE_DEFINITIONS else "SALAD"
