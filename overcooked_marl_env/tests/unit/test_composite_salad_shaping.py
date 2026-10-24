import unittest
from overcooked_marl_env import OvercookedMARL
from overcooked_marl_env.core.recipes import OrderQueue

class TestCompositeSaladShaping(unittest.TestCase):
    def test_tutorial_recipe_order(self):
        """Verify GARDEN_SALAD is the first pending recipe in tutorial.txt."""
        env = OvercookedMARL(render_mode=None, map_path="tutorial.txt", num_fixed_agents=2, featurized_obs=True)
        obs, _ = env.reset()
        pending = OrderQueue.get_pending(env.state.get("recipe_order", []), [])
        self.assertGreater(len(pending), 0)
        self.assertEqual(pending[0], "GARDEN_SALAD", "1st recipe in tutorial.txt must be GARDEN_SALAD to break simple salad local optimum")
        env.close()

    def test_2_and_4_agents_obs_shape(self):
        """Verify 2 and 4 agents initialize correctly on tutorial.txt with shape (150,)."""
        for n in [2, 4]:
            env = OvercookedMARL(render_mode=None, map_path="tutorial.txt", num_fixed_agents=n, featurized_obs=True)
            obs, _ = env.reset()
            self.assertEqual(len(obs), n)
            for a_id, a_obs in obs.items():
                self.assertEqual(a_obs.shape, (150,), f"Agent {a_id} observation shape must be (150,)")
            env.close()

if __name__ == "__main__":
    unittest.main()
