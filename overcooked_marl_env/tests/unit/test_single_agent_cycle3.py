"""
Unit tests for Cycle 3: Single-Agent Recipe Delivery & Milestone Reward Shaping.
"""

import unittest
import numpy as np
from overcooked_marl_env.env import OvercookedMARL
from overcooked_marl_env.core.items import ItemType, ItemState, create_ingredient, create_plate, add_ingredient_to_plate
from overcooked_marl_env.mechanics.reward_shaper import RewardValues

class TestSingleAgentCycle3(unittest.TestCase):
    """Test suite verifying Cycle 3 milestone reward shaping and single-agent delivery."""

    def setUp(self):
        self.env = OvercookedMARL(
            render_mode=None,
            num_fixed_agents=1,
            featurized_obs=True
        )
        self.obs, self.info = self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_featurized_observation_shape_and_dynamic_hud(self):
        """Test observation vector is strictly (150,) and HUD advances upon delivery."""
        agent_id = "agent_0"
        self.assertEqual(self.obs[agent_id].shape, (150,))

        # Initial HUD: rec_1 should be SIMPLE_SALAD (1 ingredient: lettuce), rec_2 GARDEN_SALAD (2 ingredients)
        # HUD starts at index 80: rec_1 at 80:90, rec_2 at 90:100
        rec1_total_req = self.obs[agent_id][80]
        rec2_total_req = self.obs[agent_id][90]
        self.assertAlmostEqual(rec1_total_req, 1.0 / 4.0, places=3)  # SIMPLE_SALAD: 1 ing
        self.assertAlmostEqual(rec2_total_req, 2.0 / 4.0, places=3)  # GARDEN_SALAD: 2 ings

        # Simulate delivery of SIMPLE_SALAD
        self.env.state["delivered_recipes"].append("SIMPLE_SALAD")
        self.env.state["dishes_pending"] -= 1

        obs = self.env._get_obs()
        rec1_updated_total = obs[agent_id][80]
        rec2_updated_total = obs[agent_id][90]

        # HUD must now reflect GARDEN_SALAD (2 ings) as first pending order and FRESH_SALAD (3 ings) as second
        self.assertAlmostEqual(rec1_updated_total, 2.0 / 4.0, places=3)  # GARDEN_SALAD
        self.assertAlmostEqual(rec2_updated_total, 3.0 / 4.0, places=3)  # FRESH_SALAD

    def test_chopping_board_visual_dimension(self):
        """Test dimension 30 in facing_state shows cutting progress of front item even when hands are empty."""
        agent_id = "agent_0"
        # Place agent facing cutting board at (3, 1) from (3, 2)
        self.env.state["agent_pos"][agent_id] = (3, 2)
        self.env.state["agent_orientation"][agent_id] = (0, -1)  # Facing North -> (3, 1) is 'C'

        # Put a raw tomato with 3 cuts needed on chopping board (3, 1)
        tomato = create_ingredient("tomato")
        tomato["cuts_completed"] = 1
        tomato["cuts_needed"] = 3
        t_id = self.env.interaction_handler._create_item_instance(tomato, self.env.state)
        self.env.state["ground_instances"][(3, 1)] = t_id

        obs = self.env._get_obs()
        # Front cuts progress is at index 30 in facing_state (24 + 6 = 30)
        self.assertAlmostEqual(obs[agent_id][30], 1.0 / 3.0, places=4)

        # Update cuts to 3 (ready)
        tomato["cuts_completed"] = 3
        self.env.state["item_instances"][t_id] = tomato
        obs = self.env._get_obs()
        self.assertAlmostEqual(obs[agent_id][30], 1.0, places=4)

    def test_milestone_delivery_and_episode_victory(self):
        """Test delivery rewards, order bonus, and episode victory bonus on completing all dishes."""
        agent_id = "agent_0"
        initial_dishes = self.env.state["dishes_pending"]
        self.assertGreaterEqual(initial_dishes, 3)

        # Create a plate with lettuce (Simple Salad)
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "lettuce")
        plate_id = self.env.interaction_handler._create_item_instance(plate, self.env.state)
        self.env.state["agent_holding_instance"][agent_id] = plate_id

        # Place agent in front of delivery station D (6, 3) from (5, 3) facing East
        self.env.state["agent_pos"][agent_id] = (5, 3)
        self.env.state["agent_orientation"][agent_id] = (1, 0)  # Facing East -> (6, 3) is 'D'

        # Deliver 1st recipe (SIMPLE_SALAD)
        actions = {agent_id: 4}  # Interact
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # Check reward: base recipe reward (40.0)
        self.assertGreater(rewards[agent_id], 20.0)
        self.assertEqual(self.env.state["dishes_pending"], initial_dishes - 1)
        self.assertFalse(terminations[agent_id])

        # Deliver 2nd recipe (GARDEN_SALAD)
        plate2 = create_plate()
        plate2 = add_ingredient_to_plate(plate2, "lettuce")
        plate2 = add_ingredient_to_plate(plate2, "tomato")
        plate2_id = self.env.interaction_handler._create_item_instance(plate2, self.env.state)
        self.env.state["agent_holding_instance"][agent_id] = plate2_id
        self.env.state["agent_pos"][agent_id] = (5, 3)
        self.env.state["agent_orientation"][agent_id] = (1, 0)

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.assertGreater(rewards[agent_id], 20.0)
        self.assertEqual(self.env.state["dishes_pending"], initial_dishes - 2)
        self.assertFalse(terminations[agent_id])

        # Deliver 3rd recipe (FRESH_SALAD)
        plate3 = create_plate()
        plate3 = add_ingredient_to_plate(plate3, "lettuce")
        plate3 = add_ingredient_to_plate(plate3, "tomato")
        plate3 = add_ingredient_to_plate(plate3, "cucumber")
        plate3_id = self.env.interaction_handler._create_item_instance(plate3, self.env.state)
        self.env.state["agent_holding_instance"][agent_id] = plate3_id
        self.env.state["agent_pos"][agent_id] = (5, 3)
        self.env.state["agent_orientation"][agent_id] = (1, 0)

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.assertGreater(rewards[agent_id], 20.0)

    def test_invalid_delivery_penalty(self):
        """Test delivering empty plate or invalid recipe gives negative penalty instead of reward."""
        agent_id = "agent_0"
        # Carrying empty plate
        plate = create_plate()
        plate_id = self.env.interaction_handler._create_item_instance(plate, self.env.state)
        self.env.state["agent_holding_instance"][agent_id] = plate_id

        self.env.state["agent_pos"][agent_id] = (5, 3)
        self.env.state["agent_orientation"][agent_id] = (1, 0)

        actions = {agent_id: 4}
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        # Penalty should be negative
        self.assertLess(rewards[agent_id], 0.0)

    def test_full_cooking_lifecycle_step_by_step(self):
        """Test complete step-by-step cooking cycle: pick lettuce -> place on board -> chop -> plate -> combine -> deliver."""
        agent_id = "agent_0"
        initial_dishes = self.env.state["dishes_pending"]
        
        # 1. Pick lettuce at dispenser (3, 5) from position (3, 4) facing South
        self.env.state["agent_pos"][agent_id] = (3, 4)
        self.env.state["agent_orientation"][agent_id] = (0, 1)
        obs, r1, _, _, _ = self.env.step({agent_id: 4})
        self.assertIn(agent_id, self.env.state["agent_holding_instance"])
        self.assertGreater(r1[agent_id], 0.0)  # Pickup needed ingredient receives positive milestone shaping


        # 2. Place on chopping board (3, 1) from position (3, 2) facing North
        self.env.state["agent_pos"][agent_id] = (3, 2)
        self.env.state["agent_orientation"][agent_id] = (0, -1)
        obs, r2, _, _, _ = self.env.step({agent_id: 4})
        self.assertNotIn(agent_id, self.env.state["agent_holding_instance"])
        self.assertIn((3, 1), self.env.state["ground_instances"])
        self.assertGreater(r2[agent_id], 5.0)  # Place on chop milestone

        # 3. Chop 3 times
        obs, r_c1, _, _, _ = self.env.step({agent_id: 4})  # cut 1
        obs, r_c2, _, _, _ = self.env.step({agent_id: 4})  # cut 2
        obs, r_c3, _, _, _ = self.env.step({agent_id: 4})  # cut 3 (completion)
        self.assertGreater(r_c3[agent_id], r_c1[agent_id])  # Completion milestone is higher than single cut
        chopped_id = self.env.state["ground_instances"][(3, 1)]
        self.assertTrue(self.env.state["item_instances"][chopped_id]["state"] == ItemState.READY)

        # 4. Pick plate at dispenser (4, 1) from position (4, 2) facing North
        self.env.state["agent_pos"][agent_id] = (4, 2)
        self.env.state["agent_orientation"][agent_id] = (0, -1)
        obs, r_p, _, _, _ = self.env.step({agent_id: 4})
        self.assertIn(agent_id, self.env.state["agent_holding_instance"])
        plate_inst_id = self.env.state["agent_holding_instance"][agent_id]
        self.assertEqual(self.env.state["item_instances"][plate_inst_id]["type"], ItemType.DISH)

        # 5. Combine plate with chopped lettuce at (3, 1) from (3, 2) facing North
        self.env.state["agent_pos"][agent_id] = (3, 2)
        self.env.state["agent_orientation"][agent_id] = (0, -1)
        obs, r_comb, _, _, _ = self.env.step({agent_id: 4})
        self.assertGreater(r_comb[agent_id], 20.0)  # Combine + plate completed bonus
        plate_item = self.env.state["item_instances"][plate_inst_id]
        self.assertIn("lettuce", plate_item["ingredients"])

        # 6. Deliver at (6, 3) from (5, 3) facing East
        self.env.state["agent_pos"][agent_id] = (5, 3)
        self.env.state["agent_orientation"][agent_id] = (1, 0)
        obs, r_del, term, trunc, _ = self.env.step({agent_id: 4})
        self.assertGreater(r_del[agent_id], 20.0)  # Delivery reward
        self.assertIn("SIMPLE_SALAD", self.env.state["delivered_recipes"])
        self.assertEqual(self.env.state["dishes_pending"], initial_dishes - 1)

        # 7. Verify next order is GARDEN_SALAD (2 ingredients)
        self.assertAlmostEqual(obs[agent_id][80], 2.0 / 4.0, places=3)


if __name__ == "__main__":
    unittest.main()

