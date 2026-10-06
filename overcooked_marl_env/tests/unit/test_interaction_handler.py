"""
Unit tests for InteractionHandler.

Tests all interaction types:
- Chopping board interactions (place, process, pickup)
- Plate dispenser interactions
- Ingredient dispenser interactions
- Counter interactions (place, pickup, combine)
- Trash interactions
- Delivery interactions
"""

import numpy as np
from unittest.mock import Mock, MagicMock
from overcooked_marl_env.tests.base import OvercookedTestCase
from overcooked_marl_env.mechanics.interaction_handler import InteractionHandler
from overcooked_marl_env.mechanics.reward_shaper import RewardValues
from overcooked_marl_env.core.items import (
    ItemType, ItemState,
    create_ingredient, create_plate,
    add_ingredient_to_plate
)


class TestInteractionHandler(OvercookedTestCase):
    """Test suite for InteractionHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock managers
        self.item_manager = Mock()
        self.recipe_manager = Mock()
        
        # Default mock returns to prevent non-iterable mock errors
        self.recipe_manager.get_recipe_by_name.return_value = None
        self.recipe_manager.validate_delivery.return_value = (False, "", 0.0)
        
        # Configure item_manager to return cell codes (matching core/items.py)
        self.item_manager.get_cell_code.side_effect = lambda x: {
            ' ': 0,
            '#': 1,   # Counter
            'I': 2,   # Ingredient dispenser
            'C': 3,   # Chopping board
            'F': 4,   # ? (not used)
            'P': 5,   # Plate dispenser
            'D': 6,   # Delivery
            'T': 7,   # Trash
        }.get(x, 0)
        
        # Configure item_manager to create ingredients
        def create_ingredient_by_name(name):
            return create_ingredient(name)
        
        self.item_manager.create_ingredient_by_name.side_effect = create_ingredient_by_name
        
        # Create legacy rewards config to match unit tests assertions
        legacy_rewards = RewardValues(
            pickup=2.0,
            place=2.0,
            place_on_chop=2.0,
            process=2.0,
            combine=4.0,
            invalid=-2.0,
            trash_penalty=2.0,
            delivery_attempt_reward=-2.0,
            order_bonus_mult=0.3,
            sequence_complete_bonus=0.0,
            delivery_proximity_reward=0.0,
            progress_pickup_needed=0.0,
            progress_place_on_chop=0.0,
            progress_pickup_ready=0.0,
            progress_combine_ready=0.0,
            progress_pickup_complete_plate=0.0,
            progress_ingredient_added_base=0.0,
            progress_ingredient_added_increment=0.0,
            progress_pickup_plate_with_ready=0.0,
            progress_ready_to_combine=0.0,
            progress_partial_plate=0.0,
            repetition_penalty=0.0,
            accumulation_penalty=0.0,
            delivery_multiplier_2nd=1.0,
            delivery_multiplier_3rd=1.0,
            delivery_multiplier_4th_plus=1.0
        )
        # Create handler
        self.handler = InteractionHandler(self.item_manager, self.recipe_manager, rewards=legacy_rewards)
        
        # Create basic game state
        self.game_state = self._create_game_state()
    
    def _create_game_state(self):
        """Create a basic game state for testing."""
        state = self.create_basic_game_state()
        state["recipe_order"] = ["salad", "salad", "salad"]
        if "agent_1" in state["agent_pos"]:
            del state["agent_pos"]["agent_1"]
        if "agent_1" in state["agent_orientation"]:
            del state["agent_orientation"]["agent_1"]
        return state
    
    def _create_item_instance(self, item_dict):
        """Helper to create an item instance in game state."""
        return self.create_item_in_state(self.game_state, item_dict)
    
    # Chopping Board Tests
    
    def test_chopping_board_place_ingredient(self):
        """Test placing an ingredient on an empty chopping board."""
        # Setup: agent carrying tomato, empty chopping board
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        # Set station type to chopping board
        self.game_state["map_static"][5, 6] = 3  # Position in front
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PLACE_REWARD
        self.assertEqual(global_reward, 0.0)
        self.assertNotIn("agent_0", self.game_state["agent_holding_instance"])
        self.assertEqual(self.game_state["ground_instances"][(6, 5)], tomato_id)
    
    def test_chopping_board_process_ingredient(self):
        """Test chopping an ingredient on the board."""
        # Setup: tomato on chopping board, agent has empty hands
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["ground_instances"][(6, 5)] = tomato_id
        
        # Set station type to chopping board
        self.game_state["map_static"][5, 6] = 3
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PROCESS_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Check that item was updated
        updated_tomato = self.game_state["item_instances"][tomato_id]
        self.assertEqual(updated_tomato["cuts_done"], 1)
    
    def test_chopping_board_pickup_ready_ingredient(self):
        """Test picking up a ready ingredient from chopping board."""
        # Setup: ready tomato on chopping board
        tomato = create_ingredient("tomato")
        tomato["state"] = ItemState.READY
        tomato["cuts_done"] = tomato["cuts_needed"]
        tomato_id = self._create_item_instance(tomato)
        self.game_state["ground_instances"][(6, 5)] = tomato_id
        
        # Set station type to chopping board
        self.game_state["map_static"][5, 6] = 3
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PICKUP_REWARD
        self.assertEqual(global_reward, 0.0)
        self.assertEqual(self.game_state["agent_holding_instance"]["agent_0"], tomato_id)
        self.assertNotIn((6, 5), self.game_state["ground_instances"])
    
    def test_chopping_board_invalid_interaction(self):
        """Test invalid chopping board interaction."""
        # Setup: agent carrying plate (not processable)
        plate = create_plate()
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        # Set station type to chopping board
        self.game_state["map_static"][5, 6] = 3
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, -2.0)  # INVALID_INTERACTION_PENALTY
        self.assertEqual(global_reward, 0.0)
    
    # Plate Dispenser Tests
    
    def test_plate_dispenser_pickup_empty_plate(self):
        """Test picking up an empty plate from dispenser."""
        # Setup: empty hands
        self.game_state["map_static"][5, 6] = 5  # Plate dispenser
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PICKUP_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Check that agent is now holding a plate
        plate_id = self.game_state["agent_holding_instance"]["agent_0"]
        plate = self.game_state["item_instances"][plate_id]
        self.assertEqual(plate["type"], ItemType.DISH)
        self.assertEqual(plate["ingredients"], [])
    
    def test_plate_dispenser_combine_with_ready_ingredient(self):
        """Test picking up plate and combining with READY ingredient."""
        # Setup: carrying ready tomato
        tomato = create_ingredient("tomato")
        tomato["state"] = ItemState.READY
        tomato["cuts_done"] = tomato["cuts_needed"]
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        self.game_state["map_static"][5, 6] = 5  # Plate dispenser
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 6.0)  # PICKUP_REWARD + COMBINE_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Check that tomato was deleted
        self.assertNotIn(tomato_id, self.game_state["item_instances"])
        
        # Check that agent is holding plate with tomato
        plate_id = self.game_state["agent_holding_instance"]["agent_0"]
        plate = self.game_state["item_instances"][plate_id]
        self.assertEqual(plate["type"], ItemType.DISH)
        self.assertIn("tomato", plate["ingredients"])
    
    def test_plate_dispenser_carrying_non_ready_ingredient(self):
        """Test invalid interaction: carrying non-ready ingredient."""
        # Setup: carrying unprocessed tomato
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        self.game_state["map_static"][5, 6] = 5  # Plate dispenser
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, -2.0)  # INVALID_INTERACTION_PENALTY
        self.assertEqual(global_reward, 0.0)
    
    # Ingredient Dispenser Tests
    
    def test_ingredient_dispenser_pickup(self):
        """Test picking up an ingredient from dispenser."""
        # Setup: dispenser at (6, 5) with tomato
        self.game_state["dispensers"][(6, 5)] = "tomato"
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PICKUP_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Check that agent is holding tomato
        tomato_id = self.game_state["agent_holding_instance"]["agent_0"]
        tomato = self.game_state["item_instances"][tomato_id]
        self.assertEqual(tomato["name"], "tomato")
    
    def test_ingredient_dispenser_hands_full(self):
        """Test cannot pick up if hands are full."""
        # Setup: already holding something
        plate = create_plate()
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["dispensers"][(6, 5)] = "lettuce"
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 0.0)  # ZERADO - without exploration penalty
        self.assertEqual(global_reward, 0.0)
    
    # Counter Tests
    
    def test_counter_place_item(self):
        """Test placing an item on an empty counter."""
        # Setup: carrying tomato, empty counter
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        self.game_state["map_static"][5, 6] = 1  # Counter (or trash - same thing)
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PLACE_REWARD
        self.assertEqual(global_reward, 0.0)
        self.assertNotIn("agent_0", self.game_state["agent_holding_instance"])
        self.assertEqual(self.game_state["ground_instances"][(6, 5)], tomato_id)
    
    def test_counter_pickup_item(self):
        """Test picking up an item from counter."""
        # Setup: lettuce on counter, empty hands
        lettuce = create_ingredient("lettuce")
        lettuce_id = self._create_item_instance(lettuce)
        self.game_state["ground_instances"][(6, 5)] = lettuce_id
        
        self.game_state["map_static"][5, 6] = 1  # Counter
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # PICKUP_REWARD
        self.assertEqual(global_reward, 0.0)
        self.assertEqual(self.game_state["agent_holding_instance"]["agent_0"], lettuce_id)
        self.assertNotIn((6, 5), self.game_state["ground_instances"])
    
    def test_counter_combine_ready_with_plate_on_counter(self):
        """Test combining READY ingredient with plate on counter."""
        # Setup: carrying ready tomato, plate on counter
        tomato = create_ingredient("tomato")
        tomato["state"] = ItemState.READY
        tomato["cuts_done"] = tomato["cuts_needed"]
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        plate = create_plate()
        plate_id = self._create_item_instance(plate)
        self.game_state["ground_instances"][(6, 5)] = plate_id
        
        self.game_state["map_static"][5, 6] = 1  # Counter
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 4.0)  # COMBINE_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Tomato should be deleted
        self.assertNotIn(tomato_id, self.game_state["item_instances"])
        
        # Plate should have tomato
        updated_plate = self.game_state["item_instances"][plate_id]
        self.assertIn("tomato", updated_plate["ingredients"])
    
    def test_counter_combine_plate_with_ready_on_counter(self):
        """Test combining plate with READY ingredient on counter."""
        # Setup: carrying plate, ready lettuce on counter
        plate = create_plate()
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        lettuce = create_ingredient("lettuce")
        lettuce["state"] = ItemState.READY
        lettuce["cuts_done"] = lettuce["cuts_needed"]
        lettuce_id = self._create_item_instance(lettuce)
        self.game_state["ground_instances"][(6, 5)] = lettuce_id
        
        self.game_state["map_static"][5, 6] = 1  # Counter
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 4.0)  # COMBINE_REWARD
        self.assertEqual(global_reward, 0.0)
        
        # Lettuce should be deleted
        self.assertNotIn(lettuce_id, self.game_state["item_instances"])
        self.assertNotIn((6, 5), self.game_state["ground_instances"])
        
        # Plate should have lettuce
        updated_plate = self.game_state["item_instances"][plate_id]
        self.assertIn("lettuce", updated_plate["ingredients"])
    
    # Trash Tests - Trash deletes items completely
    
    def test_trash_discard_item(self):
        """Test discarding an item in the trash - item is deleted."""
        # Setup: carrying tomato
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        self.game_state["map_static"][5, 6] = 7  # Trash
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 2.0)  # legacy trash_penalty is 2.0
        self.assertEqual(global_reward, 0.0)
        
        # Item should be completely deleted
        self.assertNotIn(tomato_id, self.game_state["item_instances"])
        self.assertNotIn("agent_0", self.game_state["agent_holding_instance"])
        # Item should NOT be on the ground
        self.assertNotIn((6, 5), self.game_state["ground_instances"])
    
    def test_trash_empty_hands(self):
        """Test cannot trash with empty hands."""
        # Setup: empty hands
        self.game_state["map_static"][5, 6] = 7  # Trash
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert - no penalty, just no action
        self.assertEqual(local_reward, 0.0)
        self.assertEqual(global_reward, 0.0)
    
    def test_trash_deletes_plate_with_ingredients(self):
        """Test that trash can delete even plates with ingredients and claws back shaping rewards."""
        # Setup: carrying plate with ingredients and accumulated shaping reward
        plate = create_plate()
        plate["shaping_reward_accumulated"] = 35.0
        plate = add_ingredient_to_plate(plate, "tomato")
        plate = add_ingredient_to_plate(plate, "lettuce")
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["map_static"][5, 6] = 7  # Trash
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert - Reward clawback (-35.0) + trash_penalty (2.0 in legacy rewards config)
        self.assertEqual(local_reward, -35.0 + self.handler.rewards.trash_penalty)
        
        # Plate should be completely deleted
        self.assertNotIn(plate_id, self.game_state["item_instances"])
        self.assertNotIn("agent_0", self.game_state["agent_holding_instance"])
    
    # Delivery Tests
    
    def test_delivery_valid_recipe(self):
        """Test delivering a valid recipe."""
        # Setup: carrying plate with tomato and lettuce
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "tomato")
        plate = add_ingredient_to_plate(plate, "lettuce")
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["map_static"][5, 6] = 6  # Delivery
        
        # Mock recipe validation
        self.recipe_manager.validate_delivery.return_value = (True, "salad", 10.0)
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, 13.0)  # Recipe reward + 30% order bonus
        self.assertEqual(global_reward, 13.0)  # Recipe + 30% bonus
        
        # Plate should be deleted
        self.assertNotIn(plate_id, self.game_state["item_instances"])
        self.assertNotIn("agent_0", self.game_state["agent_holding_instance"])
        
        # Check game state updates
        self.assertIn("salad", self.game_state["delivered_recipes"])
        self.assertEqual(self.game_state["dishes_pending"], 2)
    
    def test_delivery_invalid_recipe(self):
        """Test delivering an invalid recipe."""
        # Setup: carrying plate with just tomato
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "tomato")
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["map_static"][5, 6] = 6  # Delivery
        
        # Mock recipe validation
        self.recipe_manager.validate_delivery.return_value = (False, None, 0.0)
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, self.handler.rewards.invalid_delivery_penalty)
        self.assertEqual(global_reward, 0.0)
        
        # Plate should NOT be deleted
        self.assertIn(plate_id, self.game_state["item_instances"])
    
    def test_delivery_not_carrying_plate(self):
        """Test cannot deliver without carrying a plate."""
        # Setup: carrying tomato (not a plate)
        tomato = create_ingredient("tomato")
        tomato_id = self._create_item_instance(tomato)
        self.game_state["agent_holding_instance"]["agent_0"] = tomato_id
        
        self.game_state["map_static"][5, 6] = 6  # Delivery
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, self.handler.rewards.invalid_delivery_penalty)
        self.assertEqual(global_reward, 0.0)
    
    def test_delivery_empty_plate(self):
        """Test cannot deliver empty plate."""
        # Setup: carrying empty plate
        plate = create_plate()
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["map_static"][5, 6] = 6  # Delivery
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert
        self.assertEqual(local_reward, self.handler.rewards.invalid_delivery_penalty)
        self.assertEqual(global_reward, 0.0)
    
    def test_delivery_wrong_order_no_bonus(self):
        """Test delivering correct recipe but wrong order (no bonus)."""
        # Setup: carrying plate with cucumber (not first in order)
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "cucumber")
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        # Recipe order is ["salad", "salad", "salad"]
        self.game_state["recipe_order"] = ["salad", "cucumber_dish", "salad"]
        
        self.game_state["map_static"][5, 6] = 6  # Delivery
        
        # Mock recipe validation
        self.recipe_manager.validate_delivery.return_value = (True, "cucumber_dish", 8.0)
        
        # Execute
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert - accepted because cucumber_dish is in pending HUD orders, but receives base reward without order bonus
        self.assertEqual(local_reward, 8.0)
        self.assertEqual(global_reward, 8.0)

    def test_delivery_unrequested_recipe_penalized(self):
        """Test delivering a recipe NOT present in pending HUD orders is rejected with penalty."""
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "tomato")
        plate_id = self._create_item_instance(plate)
        self.game_state["agent_holding_instance"]["agent_0"] = plate_id
        
        self.game_state["recipe_order"] = ["salad", "salad"]
        self.game_state["map_static"][5, 6] = 6  # Delivery
        self.recipe_manager.validate_delivery.return_value = (True, "unrequested_dish", 10.0)
        
        local_reward, global_reward = self.handler.handle_interaction(
            "agent_0", self.game_state, verbose=False
        )
        
        # Assert - wipes all points earned for that dish (-10.0)
        self.assertEqual(local_reward, -10.0)
        self.assertEqual(global_reward, -10.0)
    
    # Helper Method Tests
    
    def test_get_tile_in_front(self):
        """Test getting the tile in front of agent."""
        # Agent at (5, 5) facing right (1, 0)
        tile = self.handler._get_tile_in_front("agent_0", self.game_state)
        self.assertEqual(tile, (6, 5))
        
        # Change orientation to face down (0, 1)
        self.game_state["agent_orientation"]["agent_0"] = (0, 1)
        tile = self.handler._get_tile_in_front("agent_0", self.game_state)
        self.assertEqual(tile, (5, 6))
        
        # Change orientation to face left (-1, 0)
        self.game_state["agent_orientation"]["agent_0"] = (-1, 0)
        tile = self.handler._get_tile_in_front("agent_0", self.game_state)
        self.assertEqual(tile, (4, 5))
        
        # Change orientation to face up (0, -1)
        self.game_state["agent_orientation"]["agent_0"] = (0, -1)
        tile = self.handler._get_tile_in_front("agent_0", self.game_state)
        self.assertEqual(tile, (5, 4))
    
    def test_is_correct_order(self):
        """Test checking if recipe is in correct order."""
        # First recipe should be correct
        is_correct = self.handler._is_correct_order("salad", self.game_state)
        self.assertTrue(is_correct)
        
        # Different recipe should be incorrect
        is_correct = self.handler._is_correct_order("cucumber_dish", self.game_state)
        self.assertFalse(is_correct)
        
        # After delivering first salad
        self.game_state["delivered_recipes"].append("salad")
        is_correct = self.handler._is_correct_order("salad", self.game_state)
        self.assertTrue(is_correct)  # Still correct (second salad)

    # Duplicate Ingredient Prevention Tests

    def test_duplicate_ingredient_lock_on_items(self):
        """Test items.add_ingredient_to_plate ignores duplicate ingredients."""
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "lettuce")
        self.assertEqual(plate["ingredients"], ["lettuce"])

        # Try adding duplicate lettuce
        plate = add_ingredient_to_plate(plate, "lettuce")
        self.assertEqual(plate["ingredients"], ["lettuce"])

        # Add different ingredient
        plate = add_ingredient_to_plate(plate, "tomato")
        self.assertEqual(plate["ingredients"], ["lettuce", "tomato"])

    def test_combine_duplicate_ingredient_on_counter_returns_invalid(self):
        """Test combining an ingredient already on a plate returns invalid reward."""
        # Set agent at (5, 5) facing up (0, -1) -> front tile is (5, 4)
        self.game_state["agent_pos"]["agent_0"] = (5, 5)
        self.game_state["agent_orientation"]["agent_0"] = (0, -1)
        tile_in_front = (5, 4)
        self.game_state["map_static"][5, 4] = 1  # Counter

        # Set up plate with lettuce on counter
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "lettuce")
        plate_id = self._create_item_instance(plate)
        self.game_state["ground_instances"][tile_in_front] = plate_id

        # Agent holds ready lettuce
        lettuce = create_ingredient("lettuce")
        lettuce["state"] = ItemState.READY
        lettuce_id = self._create_item_instance(lettuce)
        self.game_state["agent_holding_instance"]["agent_0"] = lettuce_id

        # Interact to combine
        local_reward, _ = self.handler.handle_interaction("agent_0", self.game_state, verbose=False)

        # Assert duplicate combine was rejected with invalid reward
        self.assertEqual(local_reward, self.handler.rewards.invalid)
        self.assertEqual(self.game_state["item_instances"][plate_id]["ingredients"], ["lettuce"])
        self.assertEqual(self.game_state["agent_holding_instance"]["agent_0"], lettuce_id)




if __name__ == "__main__":
    import unittest
    unittest.main()

