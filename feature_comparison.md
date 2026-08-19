# Feature Comparison: Original Overcooked vs. MARL Environment

This document maps the features of the original Overcooked game series against our current MARL python environment implementation.

### References (Fontes Consultadas)
- [Overcooked Wiki - Levels](https://overcooked.fandom.com/wiki/Category:Levels)
- [Overcooked Wiki - Recipes](https://overcooked.fandom.com/wiki/Category:Recipes)
- [Overcooked Wiki - Kitchen Utensils](https://overcooked.fandom.com/wiki/Category:Kitchen_utensils)
- [Overcooked Wiki - Ingredients](https://overcooked.fandom.com/wiki/Category:Ingredients)

---

## 1. Ingredients (Ingredientes)

| Category | Original Game (Wiki) | Implemented in Codebase | Not Implemented |
| :--- | :--- | :--- | :--- |
| **Vegetables** | Tomato, Lettuce, Onion, Cucumber, Carrot, Potato, Bok Choy, Broccoli, Cabbage, Mushroom | `tomato`, `lettuce`, `cucumber`, `onion` (dynamic) | Carrot, Potato, Bok Choy, Broccoli, Cabbage, Mushroom |
| **Meat & Proteins**| Beef/Meatball, Chicken, Pork/Bacon, Sausage/Hot Dog, Fish, Shrimp, Turkey | None | All meat, poultry, seafood, and protein items |
| **Carbs & Baking** | Rice, Bun/Bread, Tortilla, Pasta, Flour, Noodles, Pizza Dough | None | All grains, flour, doughs, buns, and wraps |
| **Fruits** | Apple, Banana, Blackberry, Blueberry, Cherry, Pineapple, Strawberry | None | All sweet fruits |
| **Dairy & Sweets** | Cheese, Butter, Egg, Milk, Chocolate/Cocoa, Marshmallow, Honey | None | All dairy, eggs, and confectionary ingredients |

---

## 2. Recipes (Receitas)

| Recipe Category | Original Game Examples | Implemented in Codebase | Not Implemented |
| :--- | :--- | :--- | :--- |
| **Raw/Cold Dishes** | Salads, Fruit Platters, Sushi, Sashimi | `SIMPLE_SALAD`, `GARDEN_SALAD`, `FRESH_SALAD`, `ONION_SALAD` (dynamic) | Fruit Platters, Sushi, Sashimi |
| **Cooked Soups** | Onion Soup, Tomato Soup, Mushroom Soup | `ONION_SOUP` (dynamic) | Tomato Soup, Mushroom Soup, Leek Soup |
| **Grill/Pan Meals** | Burgers, Steaks, Kebabs, Hot Dogs | None | All burgers, steaks, hot dogs, and kebabs |
| **Baking / Ovens** | Pizzas, Cakes, Pancakes, Pies, S'mores | None | All pizzas, cakes, pancakes, and baking recipes |
| **Steamed Meals** | Steamed Fish, Steamed Meat, Steamed Buns | None | All steaming-station recipes |
| **Deep Fried** | Fish & Chips, Chicken Nuggets, French Fries | None | All deep-fryer recipes |

---

## 3. Kitchen Utensils & Stations (Utensílios e Estações)

| Tile / Utensil | Original Game Function | Implemented in Codebase | Not Implemented |
| :--- | :--- | :--- | :--- |
| **Counters (`#`)** | Staging surfaces for items | **Fully Implemented**: Items can be placed, picked up, and combined on counters. | None |
| **Chopping Board (`C`)**| Processing raw ingredients | **Fully Implemented**: Supports cutting progression (1 to 3 cuts) with visual states. | None |
| **Trash Bin (`T`)** | Discarding ingredients / burnt food | **Fully Implemented**: Interacting with trash clears chef inventory. | None |
| **Plate Dispenser (`P`)**| Yields clean plates | **Fully Implemented**: Provides plates to hold chopped ingredients. | Plate limit / stacks (dispenser has infinite plates in our code). |
| **Delivery (`D`)** | Delivering completed recipe plates | **Fully Implemented**: Validates dishes, awards score, and sequence bonuses. | Tips calculation, customer patience depletion. |
| **Sinks & Washing** | Washing dirty plates returned from delivery| None | Dirty plates return loop and washing mechanic. Plates are currently infinite. |
| **Pots & Pans** | Boiling soups/rice/pasta, frying meat | None | Cook timers, overcooking, fire hazard, and stove interactions. |
| **Steamers & Ovens** | Baking pizzas/cakes, steaming buns | None | Steaming/baking timers and machinery. |
| **Fryers & Mixers** | Deep frying chips, mixing cake batter | **Placeholder**: `F` (Fryer) code exists in `items.py` but has no gameplay logic. | Deep fryer and mixer mechanics. |
| **Fire Extinguisher**| Putting out fires caused by overcooking | None | Fire hazard mechanics and firefighting tools. |
| **Gizmos & Hazards**| Conveyor belts, alavancas/levers, portals | None | Moving countertops, portal doors, item throwing, or level switches. |

---

## 4. Levels & Environment Mechanics (Níveis e Dinâmicas)

| Gameplay Mechanic | Original Game Feature | Implemented in Codebase | Not Implemented |
| :--- | :--- | :--- | :--- |
| **Layouts** | Multi-room, moving kitchens, split spaces | **Fully Implemented**: Grid template parser loads custom layouts from text files. | Moving platforms/counters. |
| **Chef Throwing** | Throwing raw ingredients to chefs or counters | None | Chefs must walk to hand over items; throwing is not supported. |
| **RL Environment** | PettingZoo/OpenAI Gym compatibility | **Fully Implemented**: PettingZoo Parallel Env API with high performance. | None |
| **Multiplayer Controls**| Split-keyboard controls for human playtests | **Fully Implemented**: Multi-agent same-keyboard setup (`keyboard_play.py`). | Online multiplayer, gamepad support. |
| **HUD / UI** | Recipe cards, timers, scores, chef status | **Fully Implemented**: Cozy Cappuccino aesthetic with custom recipe cards and score box. | Recipe patience bars/clocks on cards. |
| **Environmental Hazards**| Slippery ice, darkness, ghosts, fires | None | All environmental hazards and map-specific rules. |
