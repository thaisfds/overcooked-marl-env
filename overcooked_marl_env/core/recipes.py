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

# Definições de receitas disponíveis no ambiente
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
    # Para adicionar novas receitas, basta adicioná-las aqui seguindo o mesmo formato
}

class RecipeManager:
    def __init__(self, item_manager):
        self.item_manager = item_manager
        self.recipes = {}
        self.FINAL_TO_BIT = {ItemType.DISH: 1}
        self.ALL_FINAL_CODES = []

    def add_recipe(self, recipe: Recipe):
        """Adiciona uma nova receita e atualiza o mapeamento bit-wise"""
        self.recipes[recipe.type] = recipe
        bit_value = 2
        for ingredient in recipe.ingredients:
            if ingredient not in self.FINAL_TO_BIT:
                self.FINAL_TO_BIT[ingredient] = bit_value
                bit_value *= 2
                self.ALL_FINAL_CODES.append(ingredient)

    def get_recipe_code(self, recipe_name: str) -> int:
        """Calcula o código final esperado para uma receita."""
        recipe = self.recipes.get(recipe_name.upper())
        if not recipe:
            return 0
            
        code = self.FINAL_TO_BIT[ItemType.DISH]  # Começa com prato (1)
        for ingredient in recipe.ingredients:
            code += self.FINAL_TO_BIT[ingredient]
        return code

    def is_final_ingredient(self, item_code) -> bool:
        """Verifica se o item é um ingrediente final válido."""
        return item_code in self.ALL_FINAL_CODES

    def get_combination_value(self, base_code: int, ingredient_code: int) -> int:
        """Combina um código base com um novo ingrediente."""
        bit_value = self.FINAL_TO_BIT.get(ingredient_code, 0)
        return base_code + bit_value


# Funções auxiliares para gerenciamento de receitas
def get_available_recipes() -> List[str]:
    """Retorna uma lista com os nomes de todas as receitas disponíveis"""
    return list(RECIPE_DEFINITIONS.keys())

def get_default_recipe() -> str:
    """Retorna o nome da receita padrão"""
    return list(RECIPE_DEFINITIONS.keys())[0] if RECIPE_DEFINITIONS else "SALAD"
