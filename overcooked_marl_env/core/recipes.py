from typing import List, Optional, Tuple, Dict, Any

class OrderQueue:
    """
    Centralized manager for recipe order queues.
    Eliminates duplicated pending recipes calculation across the codebase.
    """
    @staticmethod
    def get_pending(recipe_order: Optional[List[str]], delivered_recipes: Optional[List[str]]) -> List[str]:
        """
        Returns list of remaining uncompleted recipes in FIFO order.
        
        Args:
            recipe_order: Full sequence of requested recipe names
            delivered_recipes: Sequence of already delivered recipe names
            
        Returns:
            List of pending recipes awaiting delivery
        """
        if not recipe_order:
            return []
        pending = list(recipe_order)
        for delivered in (delivered_recipes or []):
            if delivered in pending:
                pending.remove(delivered)
        return pending

    @staticmethod
    def get_next_pending(recipe_order: Optional[List[str]], delivered_recipes: Optional[List[str]]) -> Optional[str]:
        """Returns the next pending recipe at the head of the queue, or None if all delivered."""
        pending = OrderQueue.get_pending(recipe_order, delivered_recipes)
        return pending[0] if pending else None


class Recipe:
    """Base class for all recipes."""
    
    def __init__(self, name, ingredients, reward):
        """
        Initialize a recipe.
        
        Args:
            name: Recipe name (e.g., "SALAD", "BURGER")
            ingredients: List of ingredient names required (e.g., ["tomato", "lettuce"])
            reward: Points awarded for delivering this recipe
        """
        self.name = name
        self.ingredients = sorted([i.lower() for i in ingredients])  # Sort for consistent matching
        self.reward = reward
    
    def matches(self, ingredient_list):
        """
        Check if a list of ingredients matches this recipe exactly.
        
        Args:
            ingredient_list: List of ingredient names
        
        Returns:
            True if ingredients match exactly, False otherwise
        """
        return sorted([i.lower() for i in ingredient_list]) == self.ingredients
    
    def __repr__(self):
        return f"Recipe({self.name}, {self.ingredients}, reward={self.reward})"


class RecipeManager:
    """Manages all available recipes and validates combinations."""
    
    # Class-level list of custom recipes to share across all instances
    _custom_recipes = []
    
    def __init__(self):
        """Initialize recipe manager with available recipes."""
        self.recipes = []
        self._register_default_recipes()
        # Add globally registered custom recipes
        for recipe in RecipeManager._custom_recipes:
            self.register_recipe(recipe)
    
    def _register_default_recipes(self):
        """Register all default recipes, strictly loaded from default_recipes.json."""
        import os
        import json
        json_path = os.path.join(os.path.dirname(__file__), "default_recipes.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recipes = data.get("recipes", data) if isinstance(data, dict) else data
        if isinstance(recipes, list):
            self.load_recipes_from_dict(recipes)
    
    @classmethod
    def register_custom_recipe_globally(cls, recipe):
        """Register a custom recipe globally so all new instances will include it."""
        cls._custom_recipes = [r for r in cls._custom_recipes if r.name != recipe.name]
        cls._custom_recipes.append(recipe)
        
    def register_recipe(self, recipe):
        """
        Register a new recipe.
        
        Args:
            recipe: Recipe instance to register
        """
        self.recipes = [r for r in self.recipes if r.name != recipe.name]
        self.recipes.append(recipe)
        
    def load_recipes_from_dict(self, recipes_list: list):
        """
        Load recipes from a list of dictionaries.
        
        Args:
            recipes_list: List of dicts, e.g., [{"name": "SOUP", "ingredients": ["tomato"], "reward": 100}]
        """
        for item in recipes_list:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            ingredients = item.get("ingredients")
            reward = item.get("reward", 100.0)
            
            if name and ingredients:
                name_upper = name.strip().upper()
                recipe_instance = Recipe(name=name_upper, ingredients=ingredients, reward=float(reward))
                
                # Register locally
                self.register_recipe(recipe_instance)
                
                # Register globally so validators see it
                RecipeManager.register_custom_recipe_globally(recipe_instance)

    def load_recipes_from_json(self, file_path: str):
        """
        Load custom recipes and ingredients from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        import json
        from .items import register_custom_ingredient
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # 1. Parse and register custom ingredients first
                if "ingredients" in data and isinstance(data["ingredients"], list):
                    for ing in data["ingredients"]:
                        if isinstance(ing, dict) and "name" in ing:
                            register_custom_ingredient(
                                name=ing["name"],
                                color=ing.get("color"),
                                cuts_needed=ing.get("cuts_needed", 3),
                                char=ing.get("char", None),
                                state_chars=ing.get("state_chars", None),
                                stages=ing.get("stages", None)
                            )
                
                # 2. Parse and register recipes
                if "recipes" in data and isinstance(data["recipes"], list):
                    self.load_recipes_from_dict(data["recipes"])
                elif "name" in data and "ingredients" in data:
                    self.load_recipes_from_dict([data])
                    
            elif isinstance(data, list):
                # Simple format: list of recipes
                self.load_recipes_from_dict(data)
                
        except Exception as e:
            print(f"Error loading custom recipes from {file_path}: {e}")
    
    def find_matching_recipe(self, ingredient_list):
        """
        Find a recipe that exactly matches the given ingredients.
        
        Args:
            ingredient_list: List of ingredient names
        
        Returns:
            Recipe instance if found, None otherwise
        """
        for recipe in self.recipes:
            if recipe.matches(ingredient_list):
                return recipe
        return None
    
    def get_recipe_by_name(self, name):
        """
        Get a recipe by its name.
        
        Args:
            name: Recipe name (case-insensitive)
        
        Returns:
            Recipe instance if found, None otherwise
        """
        name_upper = name.upper()
        for recipe in self.recipes:
            if recipe.name == name_upper:
                return recipe
        return None
    
    def get_recipe_index(self, name: str) -> int:
        """
        Get 1-based index of a recipe by name.
        Returns 0 if not found.
        
        Args:
            name: Recipe name (case-insensitive)
        """
        if not name:
            return 0
        name_upper = name.upper()
        for idx, recipe in enumerate(self.recipes, start=1):
            if recipe.name == name_upper:
                return idx
        return 0

    def get_recipe_ingredients_map(self) -> dict:
        """
        Get dictionary mapping recipe names to lists of required ingredients.
        """
        return {r.name: list(r.ingredients) for r in self.recipes}

    
    def get_all_recipe_names(self):
        """
        Get list of all available recipe names.
        
        Returns:
            List of recipe names
        """
        return [recipe.name for recipe in self.recipes]
    
    def load_recipes_from_names(self, recipe_names):
        """
        Filter recipes to only include specified names.
        Useful for loading recipes from map files.
        
        Args:
            recipe_names: List of recipe names to include
        
        Returns:
            List of Recipe instances that match the names
        """
        loaded = []
        for name in recipe_names:
            recipe = self.get_recipe_by_name(name)
            if recipe:
                loaded.append(recipe)
        return loaded
    
    def validate_delivery(self, ingredient_list):
        """
        Validate if ingredients form a valid recipe and return details.
        
        Args:
            ingredient_list: List of ingredient names
        
        Returns:
            Tuple of (is_valid, recipe_name, reward)
        """
        recipe = self.find_matching_recipe(ingredient_list)
        if recipe:
            return (True, recipe.name, recipe.reward)
        return (False, None, 0.0)


# Utility and backwards compatibility functions
def get_available_recipes() -> List[str]:
    """Get list of all available recipe names from central configuration."""
    return RecipeManager().get_all_recipe_names()


def get_default_recipe() -> str:
    """Get the default recipe name from central configuration."""
    names = get_available_recipes()
    return names[0] if names else "SIMPLE_SALAD"


def get_recipe_definitions() -> Dict[str, Recipe]:
    """Get dictionary of recipes loaded strictly from central configuration."""
    return {r.name: r for r in RecipeManager().recipes}


# Backwards compatibility dictionary populated strictly from default_recipes.json
RECIPE_DEFINITIONS = get_recipe_definitions()
