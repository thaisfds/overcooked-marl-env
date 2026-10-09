"""
Unit tests for Extensibility and Architecture enhancements:
- OrderQueue FIFO calculations
- Multi-stage ingredient processing pipelines
- Characterless and instant ingredients
- StationRegistry dynamic strategy dispatch
- Declarative Recipe & Ingredient loading
"""

import unittest
import numpy as np
from overcooked_marl_env.core.recipes import OrderQueue, Recipe, RecipeManager
from overcooked_marl_env.core.items import (
    ItemManager, register_tile, register_custom_ingredient, create_ingredient, can_process,
    advance_processing, is_ready, is_plate, create_plate,
    add_ingredient_to_plate, ItemState, ItemType
)
from overcooked_marl_env.mechanics.interaction_handler import InteractionHandler
from overcooked_marl_env.mechanics.interactions.base_station import StationHandler
from overcooked_marl_env.mechanics.interactions.chopping_handler import ChoppingBoardHandler
from overcooked_marl_env.mechanics.interactions.station_registry import StationRegistry


class TestExtensibilityArchitecture(unittest.TestCase):
    """Test suite verifying modular architecture, multi-stage pipelines, and registries."""

    def test_order_queue_fifo(self):
        """Test OrderQueue deterministic FIFO pending logic."""
        recipe_order = ["SALAD", "BURGER", "SALAD", "PIZZA"]
        delivered = ["SALAD"]
        pending = OrderQueue.get_pending(recipe_order, delivered)
        self.assertEqual(pending, ["BURGER", "SALAD", "PIZZA"])
        self.assertEqual(OrderQueue.get_next_pending(recipe_order, delivered), "BURGER")

        # Deliver burger and second salad
        delivered.extend(["BURGER", "SALAD"])
        self.assertEqual(OrderQueue.get_pending(recipe_order, delivered), ["PIZZA"])
        self.assertEqual(OrderQueue.get_next_pending(recipe_order, delivered), "PIZZA")

        # Deliver all
        delivered.append("PIZZA")
        self.assertEqual(OrderQueue.get_pending(recipe_order, delivered), [])
        self.assertIsNone(OrderQueue.get_next_pending(recipe_order, delivered))

    def test_is_plate_helper(self):
        """Test is_plate centralized helper."""
        self.assertFalse(is_plate(None))
        self.assertFalse(is_plate({}))
        self.assertFalse(is_plate({"name": "tomato", "type": ItemType.INGREDIENT}))
        self.assertTrue(is_plate(create_plate()))
        self.assertTrue(is_plate({"name": "plate", "type": ItemType.DISH}))
        self.assertTrue(is_plate({"name": "fresh_plate", "type": "any"}))

    def test_instant_ingredient(self):
        """Test ingredient that requires 0 cuts (instant ready, e.g. bread bun, cheese)."""
        register_custom_ingredient(
            name="bread_bun",
            color=(210, 180, 140),
            cuts_needed=0
        )
        bun = create_ingredient("bread_bun")
        self.assertTrue(is_ready(bun))
        self.assertTrue(bun["combinable"])
        self.assertFalse(can_process(bun, station="C"))

    def test_multi_stage_pipeline(self):
        """Test multi-stage processing: meat chopped on 'C', then cooked on 'S'."""
        register_custom_ingredient(
            name="patty_meat",
            color=(139, 69, 19),
            stages=[
                {"station": "C", "action": "chop", "steps": 2},
                {"station": "S", "action": "cook", "steps": 2}
            ]
        )
        meat = create_ingredient("patty_meat")
        self.assertFalse(is_ready(meat))

        # At chopping board 'C': can process
        self.assertTrue(can_process(meat, station="C"))
        self.assertFalse(can_process(meat, station="S"))

        # Advance stage 1 step 1
        meat = advance_processing(meat, station="C")
        self.assertFalse(is_ready(meat))
        self.assertTrue(can_process(meat, station="C"))

        # Advance stage 1 step 2 -> completes chopping!
        meat = advance_processing(meat, station="C")
        self.assertFalse(is_ready(meat))
        self.assertFalse(can_process(meat, station="C"))  # Cannot chop anymore!
        self.assertTrue(can_process(meat, station="S"))   # Now ready for stove 'S'!

        # At stove 'S': advance step 1
        meat = advance_processing(meat, station="S")
        self.assertFalse(is_ready(meat))
        self.assertTrue(can_process(meat, station="S"))

        # At stove 'S': advance step 2 -> completes cooking!
        meat = advance_processing(meat, station="S")
        self.assertTrue(is_ready(meat))
        self.assertTrue(meat["combinable"])
        self.assertFalse(can_process(meat, station="S"))

        # Plate assembly
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, meat["name"], meat)
        self.assertIn("patty_meat", plate["ingredients"])

    def test_characterless_ingredient(self):
        """Test ingredient registered without explicit ASCII char."""
        register_custom_ingredient(
            name="truffle_oil",
            color=(100, 100, 50),
            cuts_needed=0
        )
        oil = create_ingredient("truffle_oil")
        self.assertEqual(oil["name"], "truffle_oil")
        self.assertTrue(is_ready(oil))

    def test_station_registry(self):
        """Test StationRegistry for custom station strategies."""
        registry = StationRegistry()

        class MockStoveHandler(StationHandler):
            def handle(self, agent_id, tile_pos, carrying_item, carrying_instance_id,
                       ground_item, ground_instance_id, game_state, verbose=False):
                return 42.0, 10.0

        mock_stove = MockStoveHandler(None)
        registry.register('S', mock_stove)

        self.assertIn('S', registry)
        self.assertNotIn('F', registry)
        self.assertEqual(registry.get('S'), mock_stove)
        self.assertIsNone(registry.get('F'))

        # Test execution
        reward, global_bonus = registry.get('S').handle(
            "agent_0", (0, 0), None, None, None, None, {}
        )
        self.assertEqual(reward, 42.0)
        self.assertEqual(global_bonus, 10.0)

    def test_ingredient_registry_defaults_loaded(self):
        """Test that default ingredients are loaded from default_ingredients.json."""
        from overcooked_marl_env.core.items import ingredient_registry, INGREDIENTS
        self.assertIn("tomato", ingredient_registry)
        self.assertIn("lettuce", ingredient_registry)
        self.assertIn("cucumber", ingredient_registry)
        self.assertIn("onion", ingredient_registry)
        self.assertEqual(ingredient_registry["tomato"]["color"], (255, 0, 0))
        # Ensure INGREDIENTS alias matches
        self.assertEqual(INGREDIENTS["tomato"], ingredient_registry["tomato"])

    def test_ingredient_registry_load_from_json(self):
        """Test loading custom ingredients from an arbitrary JSON file."""
        import tempfile
        import json
        import os
        from overcooked_marl_env.core.items import IngredientRegistry

        custom_data = {
            "ingredients": [
                {
                    "name": "dragonfruit",
                    "color": [255, 20, 147],
                    "cuts_needed": 4,
                    "char": "d"
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(custom_data, f)
            temp_path = f.name

        try:
            test_reg = IngredientRegistry()
            test_reg.load_from_json(temp_path)
            self.assertIn("dragonfruit", test_reg)
            self.assertEqual(test_reg["dragonfruit"]["cuts_needed"], 4)
            self.assertEqual(test_reg["dragonfruit"]["color"], (255, 20, 147))
        finally:
            os.remove(temp_path)


    def test_observation_builder_with_instant_and_custom_ingredients(self):
        """Test ObservationBuilder encodes instant ingredients and custom items correctly."""
        from overcooked_marl_env.environment.observation_builder import ObservationBuilder
        from overcooked_marl_env.core.items import ItemManager, create_ingredient, ItemType
        
        obs_builder = ObservationBuilder()
        item_mgr = ItemManager()
        
        # Test instant ingredient processing vector
        register_custom_ingredient("parmesan", color=(255, 255, 200), cuts_needed=0)
        parmesan = create_ingredient("parmesan")
        vec = obs_builder._ingredient_to_vector(parmesan)
        self.assertEqual(vec, [0, 0, 1])  # cuts_done=0, cuts_needed=0, is_ready=1
        
        # Test code for instant ingredient (state_offset should be 4 = READY)
        code = obs_builder._item_to_code(parmesan)
        self.assertGreaterEqual(code, 50)  # 10 + idx + 40 >= 50

    def test_featurized_observation_custom_stations_and_instant_ingredients(self):
        """Test FeaturizedObservationBuilder perceives custom stations and instant ingredient progress."""
        from overcooked_marl_env.environment.featurized_observation_builder import FeaturizedObservationBuilder
        from overcooked_marl_env.core.items import ItemManager, create_ingredient
        
        f_builder = FeaturizedObservationBuilder()
        item_mgr = ItemManager()
        item_mgr.register_tile('G', 12, "Grill", (200, 50, 0))
        
        # Test _get_item_cuts_progress
        raw_tomato = create_ingredient("tomato")
        self.assertEqual(f_builder._get_item_cuts_progress(raw_tomato), 0.0)
        
        register_custom_ingredient("mayo", color=(255, 255, 220), cuts_needed=0)
        mayo = create_ingredient("mayo")
        self.assertEqual(f_builder._get_item_cuts_progress(mayo), 1.0)
        
        # Test facing custom station 'G'
        game_state = {
            "map_static": np.array([[12, 0]], dtype=np.uint8),
            "agent_pos": {"agent_0": (1, 0)},
            "agent_orientation": {"agent_0": (-1, 0)},
            "agent_holding_instance": {},
            "ground_instances": {},
            "item_instances": {},
            "dispensers": {},
            "recipe_order": ["SIMPLE_SALAD"],
            "delivered_recipes": [],
            "time_steps": 0,
        }
        obs = f_builder.build_observations(game_state, ["agent_0"], item_mgr)
        # Facing state dim 0 (station norm) for custom station 'G' should be 0.4
        self.assertAlmostEqual(obs["agent_0"][24], 0.4, places=2)

    def test_observation_builder_order_channel_progression(self):
        """Test that ObservationBuilder dynamically highlights the true next pending recipe."""
        from overcooked_marl_env.environment.observation_builder import ObservationBuilder
        from overcooked_marl_env.core.recipes import RecipeManager
        
        obs_builder = ObservationBuilder()
        recipe_mgr = RecipeManager()
        
        game_state = {
            "recipe_order": ["SIMPLE_SALAD", "GARDEN_SALAD"],
            "delivered_recipes": ["SIMPLE_SALAD"],  # SIMPLE_SALAD is delivered!
            "expected_recipe_index": 0,  # Not updated manually
        }
        channel = obs_builder._build_recipe_order_channel(game_state, (5, 5), recipe_mgr)
        # Slot 0 was SIMPLE_SALAD -> status 200 (delivered)
        slot0_code = channel[0, 0]
        self.assertGreaterEqual(slot0_code, 200)
        # Slot 1 is GARDEN_SALAD -> should be next (status 100)
        slot1_code = channel[0, 1]
        self.assertTrue(100 <= slot1_code < 200, f"Expected status 100 for slot 1, got {slot1_code}")


    def test_map_loader_custom_recipes_and_stations(self):
        """Test MapLoader and MapValidator accept custom stations and custom recipes."""
        import tempfile
        import shutil
        import json
        import os
        from overcooked_marl_env.map.map_loader import MapLoader
        from overcooked_marl_env.map.map_parser import MapParser
        from overcooked_marl_env.core.recipes import RecipeManager, Recipe
        from overcooked_marl_env.core.items import register_tile

        # Register custom station tile 'S' (Stove)
        register_tile('S', 15, "Stove", (220, 80, 20))

        # Check MapParser parses 'S'
        parser = MapParser()
        grid, _ = parser.parse_grid_string("S#0\n# 1\n###")
        self.assertEqual(grid[0, 0], 15)

        # Create temporary map with custom recipe BURGER
        temp_dir = tempfile.mkdtemp()
        try:
            map_file = os.path.join(temp_dir, "custom_map.txt")
            with open(map_file, "w") as f:
                f.write("""# Grid
____
|S0|
|#1|
____

# Dispensers
meat 1 1

# Recipes
BURGER
""")
            recipes_file = os.path.join(temp_dir, "recipes.json")
            with open(recipes_file, "w") as f:
                json.dump({
                    "recipes": [
                        {"name": "BURGER", "ingredients": ["meat"], "reward": 50.0}
                    ]
                }, f)

            # MapLoader should auto-load adjacent recipes.json and pass strict validation
            loader = MapLoader(strict_validation=True)
            map_data = loader.load_map(map_file, use_default_on_error=False)
            self.assertIn("BURGER", map_data.recipes)
            self.assertEqual(map_data.static_map[1, 1], 15)  # Stove code 15
        finally:
            shutil.rmtree(temp_dir)


    def test_universal_appliance_handler_reuse(self):
        """Test reusing ChoppingBoardHandler for a custom Stove ('S') station."""
        from overcooked_marl_env.mechanics.interaction_handler import InteractionHandler
        from overcooked_marl_env.mechanics.interactions.chopping_handler import ChoppingBoardHandler
        from overcooked_marl_env.core.items import ItemManager, register_tile
        from overcooked_marl_env.core.recipes import RecipeManager

        item_mgr = ItemManager()
        recipe_mgr = RecipeManager()
        handler = InteractionHandler(item_mgr, recipe_mgr)

        # Register 'S' as station
        register_tile('S', 20, "Stove", (255, 120, 50))
        stove_handler = ChoppingBoardHandler(handler, station_char='S')
        handler.register_station_handler('S', stove_handler)

        # Ingredient with cooking at 'S'
        register_custom_ingredient(
            name="pancake_batter",
            color=(250, 240, 190),
            stages=[{"station": "S", "action": "cook", "steps": 2}]
        )
        batter = create_ingredient("pancake_batter")
        b_id = handler._create_item_instance(batter, {"next_item_id": 0, "item_instances": {}})

        game_state = {
            "map_static": np.array([[20, 0]], dtype=np.uint8),  # tile (0, 0) is 'S'
            "agent_pos": {"agent_0": (1, 0)},
            "agent_orientation": {"agent_0": (-1, 0)},  # facing (0, 0)
            "agent_holding_instance": {"agent_0": b_id},
            "ground_instances": {},
            "item_instances": {b_id: batter},
            "dispensers": {},
            "recipe_order": ["SIMPLE_SALAD"],
            "delivered_recipes": [],
            "next_item_id": 1,
        }

        # Step 1: Place on stove 'S'
        r1, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreater(r1, 0.0)
        self.assertIn((0, 0), game_state["ground_instances"])
        self.assertNotIn("agent_0", game_state["agent_holding_instance"])

        # Step 2: Cook 1/2
        r2, _ = handler.handle_interaction("agent_0", game_state)
        self.assertEqual(r2, handler.rewards.chop_cut)
        ground_batter = game_state["item_instances"][b_id]
        self.assertFalse(is_ready(ground_batter))

        # Step 3: Cook 2/2 -> Complete milestone!
        r3, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreaterEqual(r3, handler.rewards.chop_completed)
        self.assertTrue(is_ready(ground_batter))

    def test_multistage_pipeline_across_different_stations(self):
        """Test full pipeline: chop on 'C' -> pickup -> place on 'S' -> cook -> ready -> combine."""
        item_mgr = ItemManager()
        recipe_mgr = RecipeManager()
        handler = InteractionHandler(item_mgr, recipe_mgr)

        item_mgr.register_tile('S', code=25, name="Stove", color=(200, 50, 50))
        stove_handler = ChoppingBoardHandler(handler, station_char='S')
        handler.register_station_handler('S', stove_handler)

        register_custom_ingredient(
            name="burger_patty",
            color=(180, 50, 50),
            stages=[
                {"station": "C", "action": "chop", "steps": 2},
                {"station": "S", "action": "cook", "steps": 2}
            ]
        )
        patty = create_ingredient("burger_patty")
        patty_id = handler._create_item_instance(patty, {"next_item_id": 0, "item_instances": {}})

        c_code = item_mgr.get_cell_code('C')
        s_code = item_mgr.get_cell_code('S')
        # Map: tile (0, 0) is 'C', tile (2, 0) is 'S'
        map_static = np.zeros((1, 3), dtype=np.uint8)
        map_static[0, 0] = c_code
        map_static[0, 2] = s_code

        game_state = {
            "map_static": map_static,
            "agent_pos": {"agent_0": (1, 0)},
            "agent_orientation": {"agent_0": (-1, 0)},  # facing (0, 0) -> 'C'
            "agent_holding_instance": {"agent_0": patty_id},
            "ground_instances": {},
            "item_instances": {patty_id: patty},
            "dispensers": {},
            "recipe_order": [],
            "delivered_recipes": [],
            "next_item_id": 1,
        }

        # 1. Place patty on 'C'
        r_place_c, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreater(r_place_c, 0.0)
        self.assertNotIn("agent_0", game_state["agent_holding_instance"])
        self.assertIn((0, 0), game_state["ground_instances"])

        # 2. Chop 1/2 on 'C'
        r_chop1, _ = handler.handle_interaction("agent_0", game_state)
        self.assertEqual(r_chop1, handler.rewards.chop_cut)

        # 3. Chop 2/2 on 'C' -> Stage 0 completes, milestone awarded
        r_chop2, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreaterEqual(r_chop2, handler.rewards.chop_completed)
        self.assertFalse(is_ready(patty))  # Not fully ready (needs stove)

        # 4. Agent picks up chopped patty from 'C'
        r_pickup_c, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreaterEqual(r_pickup_c, 0.0)
        self.assertEqual(game_state["agent_holding_instance"]["agent_0"], patty_id)
        self.assertNotIn((0, 0), game_state["ground_instances"])

        # 5. Agent turns to face 'S' at (2, 0)
        game_state["agent_orientation"]["agent_0"] = (1, 0)

        # 6. Place on 'S'
        r_place_s, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreater(r_place_s, 0.0)
        self.assertIn((2, 0), game_state["ground_instances"])
        self.assertNotIn("agent_0", game_state["agent_holding_instance"])

        # 7. Cook 1/2 on 'S'
        r_cook1, _ = handler.handle_interaction("agent_0", game_state)
        self.assertEqual(r_cook1, handler.rewards.chop_cut)

        # 8. Cook 2/2 on 'S' -> Stage 1 completes, fully READY!
        r_cook2, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreaterEqual(r_cook2, handler.rewards.chop_completed)
        ground_patty = game_state["item_instances"][patty_id]
        self.assertTrue(is_ready(ground_patty))

        # 9. Pick up ready patty from 'S'
        r_pickup_s, _ = handler.handle_interaction("agent_0", game_state)
        self.assertGreaterEqual(r_pickup_s, 0.0)
        self.assertEqual(game_state["agent_holding_instance"]["agent_0"], patty_id)

    def test_rebalanced_recipe_rewards(self):
        """Test that default recipe rewards properly incentivize complex 2- and 3-ingredient dishes."""
        recipe_mgr = RecipeManager()
        simple = recipe_mgr.get_recipe_by_name("SIMPLE_SALAD")
        garden = recipe_mgr.get_recipe_by_name("GARDEN_SALAD")
        fresh = recipe_mgr.get_recipe_by_name("FRESH_SALAD")

        self.assertEqual(simple.reward, 20.0)
        self.assertEqual(garden.reward, 60.0)
        self.assertEqual(fresh.reward, 120.0)

    def test_json_as_single_source_of_truth(self):
        """Verify that default_recipes.json is the sole source of truth and RECIPE_DEFINITIONS stays in sync."""
        from overcooked_marl_env.core.recipes import RECIPE_DEFINITIONS, get_available_recipes, get_default_recipe
        import json, os

        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "core", "default_recipes.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_names = [r["name"] for r in data["recipes"]]
        self.assertEqual(get_available_recipes(), expected_names)
        self.assertEqual(get_default_recipe(), expected_names[0])
        self.assertEqual(set(RECIPE_DEFINITIONS.keys()), set(expected_names))
        for r_data in data["recipes"]:
            r_obj = RECIPE_DEFINITIONS[r_data["name"]]
            self.assertEqual(r_obj.reward, r_data["reward"])
            self.assertEqual(r_obj.ingredients, sorted(r_data["ingredients"]))

    def test_potential_based_delivery_guidance_shaping(self):
        """Test that calculate_step_shaping correctly guides agents holding deliverable plates towards delivery."""
        item_mgr = ItemManager()
        recipe_mgr = RecipeManager()
        handler = InteractionHandler(item_mgr, recipe_mgr)

        d_code = item_mgr.get_cell_code('D')
        plate = create_plate(["lettuce"])
        p_id = handler._create_item_instance(plate, {"next_item_id": 0, "item_instances": {}})

        # Delivery station at (5, 0)
        # Agent moving from (2, 0) -> (3, 0) gets closer (dist 3 -> 2)
        # Agent moving from (3, 0) -> (2, 0) gets farther (dist 2 -> 3)
        game_state = {
            "map_static": np.array([[0, 0, 0, 0, 0, d_code]], dtype=np.uint8),
            "agent_pos": {"agent_0": (2, 0)},
            "agent_holding_instance": {"agent_0": p_id},
            "item_instances": {p_id: plate},
            "recipe_order": ["SIMPLE_SALAD"],
            "delivered_recipes": [],
        }

        # Step 1: Moving closer (2, 0) -> (3, 0) gives positive reward
        r_closer = handler.calculate_step_shaping("agent_0", game_state, (2, 0), (3, 0))
        self.assertEqual(r_closer, 1.0)

        # Step 2: Moving farther (3, 0) -> (2, 0) gives negative reward
        r_farther = handler.calculate_step_shaping("agent_0", game_state, (3, 0), (2, 0))
        self.assertEqual(r_farther, -1.0)

        # Net reward for round trip is exactly zero (no exploit loops)
        self.assertEqual(r_closer + r_farther, 0.0)

        # Step 3: Stationary (no movement) gives zero
        r_still = handler.calculate_step_shaping("agent_0", game_state, (2, 0), (2, 0))
        self.assertEqual(r_still, 0.0)

        # Step 4: If recipe was already delivered (not in pending queue), returns 0.0
        game_state["delivered_recipes"] = ["SIMPLE_SALAD"]
        r_delivered = handler.calculate_step_shaping("agent_0", game_state, (2, 0), (3, 0))
        self.assertEqual(r_delivered, 0.0)

    def test_dynamic_dishes_pending_in_game_state(self):
        """Verify dishes_pending matches exact length of recipe_order instead of hardcoded 3."""
        from overcooked_marl_env.environment.game_state import GameState, MapData

        for count in [1, 5, 7]:
            map_data = MapData(
                static_map=np.zeros((3, 3), dtype=np.uint8),
                agent_positions={"agent_0": (1, 1)},
                dispensers={},
                recipes=[f"RECIPE_{i}" for i in range(count)],
                size=(3, 3)
            )
            state = GameState(map_data, ["agent_0"])
            self.assertEqual(state.dishes_pending, count)

            # Test reset preserves dynamic count
            state.reset(map_data)
            self.assertEqual(state.dishes_pending, count)

    def test_clean_default_map_dispensers(self):
        """Verify default_map.txt dispensers strictly match the required recipe ingredients."""
        from overcooked_marl_env.map.map_loader import MapLoader
        loader = MapLoader()
        map_data = loader.load_map("default_map.txt")
        dispenser_ingredients = set(map_data.dispensers.values())
        self.assertEqual(dispenser_ingredients, {"tomato", "lettuce", "cucumber"})


if __name__ == "__main__":
    unittest.main()



