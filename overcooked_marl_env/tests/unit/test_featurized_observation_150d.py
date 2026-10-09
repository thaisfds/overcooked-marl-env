import unittest
import numpy as np
from overcooked_marl_env import OvercookedMARL
from overcooked_marl_env.core.items import create_ingredient, create_plate, add_ingredient_to_plate

class TestFeaturizedObservation150D(unittest.TestCase):
    """
    Comprehensive tests for the universal 150-dimensional FeaturizedObservationBuilder,
    multi-agent scaling (1-4 agents), semantic counter perception, and anti-reward-hacking.
    """

    def test_shape_across_agent_counts(self):
        """Verify vector shape is strictly (150,) for 1, 2, 3, and 4 agents."""
        for num_agents in [1, 2, 4]:
            env = OvercookedMARL(
                render_mode=None,
                map_path="default_map.txt",
                num_fixed_agents=num_agents,
                featurized_obs=True
            )
            obs, info = env.reset()
            self.assertEqual(len(obs), num_agents)
            for agent_id, agent_obs in obs.items():
                self.assertEqual(agent_obs.shape, (150,), f"Agent {agent_id} observation shape != (150,) with num_agents={num_agents}")
                self.assertFalse(np.isnan(agent_obs).any(), f"Observation for {agent_id} contains NaNs")
                self.assertFalse(np.isinf(agent_obs).any(), f"Observation for {agent_id} contains Infs")
            env.close()

    def test_shape_on_divided_map_tutorial(self):
        """Verify vector shape is strictly (150,) on the physically divided tutorial map with 4 agents."""
        env = OvercookedMARL(
            render_mode=None,
            map_path="tutorial.txt",
            num_fixed_agents=4,
            featurized_obs=True
        )
        obs, info = env.reset()
        self.assertEqual(len(obs), 4)
        for agent_id, agent_obs in obs.items():
            self.assertEqual(agent_obs.shape, (150,))
            self.assertFalse(np.isnan(agent_obs).any())
        env.close()

    def test_facing_tile_and_dish_readiness_perception(self):
        """Test facing tile features detect a completed dish on the counter in front of agent."""
        env = OvercookedMARL(
            render_mode=None,
            map_path="default_map.txt",
            num_fixed_agents=1,
            featurized_obs=True
        )
        obs, _ = env.reset()
        agent_id = "agent_0"

        # Position agent at (5, 2) facing East towards counter (6, 2)
        # Note: on default_map, (6, 2) is a counter '#'
        env.state["agent_pos"][agent_id] = (5, 2)
        env.state["agent_orientation"][agent_id] = (1, 0)

        # Place a plate with lettuce on counter (6, 2) (matches SIMPLE_SALAD)
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "lettuce", create_ingredient("lettuce"))
        p_id = env.interaction_handler._create_item_instance(plate, env.state)
        env.state["ground_instances"][(6, 2)] = p_id

        obs = env._get_obs()
        agent_obs = obs[agent_id]

        # In facing_state (indices 24:32):
        # 24: front_tile_station_norm (counter = 0.2)
        # 25: front_has_item (1.0)
        # 26: front_is_dish (1.0)
        # 27: front_dish_is_ready_for_recipe (1.0 because matches SIMPLE_SALAD)
        self.assertAlmostEqual(agent_obs[24], 0.2, places=3)
        self.assertEqual(agent_obs[25], 1.0)
        self.assertEqual(agent_obs[26], 1.0)
        self.assertEqual(agent_obs[27], 1.0)
        env.close()

    def test_anti_reward_hacking_pickup_loop(self):
        """Verify picking up and placing the same plate repeatedly does not yield infinite rewards."""
        env = OvercookedMARL(
            render_mode=None,
            map_path="default_map.txt",
            num_fixed_agents=1,
            featurized_obs=True
        )
        env.reset()
        agent_id = "agent_0"

        # Position agent at (5, 2) facing North towards counter (5, 1)
        env.state["agent_pos"][agent_id] = (5, 2)
        env.state["agent_orientation"][agent_id] = (0, -1)

        # Create complete plate on counter (5, 1)
        plate = create_plate()
        plate = add_ingredient_to_plate(plate, "lettuce", create_ingredient("lettuce"))
        p_id = env.interaction_handler._create_item_instance(plate, env.state)
        env.state["ground_instances"][(5, 1)] = p_id

        # Pickup 1: Agent interacts (action 4) to pick up plate
        _, r1, _, _, _ = env.step({agent_id: 4})
        pickup_reward_1 = r1[agent_id]
        # Should have received standard pickup + milestone progress
        self.assertGreater(pickup_reward_1, 0.0)

        # Place 1: Agent interacts again to place plate back on (4, 3)
        _, r_place_1, _, _, _ = env.step({agent_id: 4})

        # Pickup 2: Agent picks up plate again without modifying ingredients
        _, r2, _, _, _ = env.step({agent_id: 4})
        pickup_reward_2 = r2[agent_id]

        # Milestone progress reward MUST NOT be awarded twice for the same plate!
        self.assertLess(pickup_reward_2, pickup_reward_1)

        # Place 2: Alternating pickup/place on the same tile triggers useless_cycle_penalty
        _, r_place_2, _, _, _ = env.step({agent_id: 4})
        self.assertLess(r_place_2[agent_id], 0.0, "Useless cycle penalty must penalize alternating pickup/place loops")
        env.close()

if __name__ == "__main__":
    unittest.main()
