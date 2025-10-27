"""
Recipe system using Python classes for flexible recipe management.
"""

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
        self.ingredients = sorted(ingredients)  # Sort for consistent matching
        self.reward = reward
    
    def matches(self, ingredient_list):
        """
        Check if a list of ingredients matches this recipe exactly.
        
        Args:
            ingredient_list: List of ingredient names
        
        Returns:
            True if ingredients match exactly, False otherwise
        """
        return sorted(ingredient_list) == self.ingredients
    
    def __repr__(self):
        return f"Recipe({self.name}, {self.ingredients}, reward={self.reward})"


class SaladRecipe(Recipe):
    """Simple salad with tomato and lettuce."""
    
    def __init__(self):
        super().__init__(
            name="SALAD",
            ingredients=["tomato", "lettuce"],
            reward=10.0
        )


class BurgerRecipe(Recipe):
    """Burger with meat and lettuce."""
    
    def __init__(self):
        super().__init__(
            name="BURGER",
            ingredients=["meat", "lettuce"],
            reward=20.0
        )


class ComplexSaladRecipe(Recipe):
    """Complex salad with tomato, lettuce and onion."""
    
    def __init__(self):
        super().__init__(
            name="COMPLEX_SALAD",
            ingredients=["tomato", "lettuce", "onion"],
            reward=30.0
        )


class RecipeManager:
    """Manages all available recipes and validates combinations."""
    
    def __init__(self):
        """Initialize recipe manager with available recipes."""
        self.recipes = []
        self._register_default_recipes()
    
    def _register_default_recipes(self):
        """Register all default recipes."""
        self.recipes = [
            SaladRecipe(),
            BurgerRecipe(),
            ComplexSaladRecipe(),
        ]
    
    def register_recipe(self, recipe):
        """
        Register a new recipe.
        
        Args:
            recipe: Recipe instance to register
        """
        self.recipes.append(recipe)
    
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


# Legacy support for backwards compatibility
def get_available_recipes():
    """Get list of all available recipe names."""
    manager = RecipeManager()
    return manager.get_all_recipe_names()


def get_default_recipe():
    """Get the default recipe name."""
    return "SALAD"


# Recipe definitions for backwards compatibility
RECIPE_DEFINITIONS = {
    "SALAD": SaladRecipe(),
    "BURGER": BurgerRecipe(),
    "COMPLEX_SALAD": ComplexSaladRecipe(),
}
