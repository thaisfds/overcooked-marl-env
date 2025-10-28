"""
Test PettingZoo API compliance.

This test verifies that OvercookedMARL follows PettingZoo conventions:
- https://pettingzoo.farama.org/tutorials/custom_environment/
"""

import unittest
import numpy as np
from overcooked_marl_env import OvercookedMARL


class TestPettingZooAPI(unittest.TestCase):
    """Test PettingZoo API compliance."""
    
    def setUp(self):
        """Create environment before each test."""
        self.env = OvercookedMARL(render_mode=None, verbose=False)
    
    def tearDown(self):
        """Clean up after each test."""
        self.env.close()
    
    def test_metadata_exists(self):
        """Test that metadata attribute exists."""
        self.assertTrue(hasattr(self.env, 'metadata'))
        self.assertIsInstance(self.env.metadata, dict)
    
    def test_metadata_has_render_modes(self):
        """Test that metadata contains render_modes."""
        self.assertIn('render_modes', self.env.metadata)
        self.assertIn('human', self.env.metadata['render_modes'])
        self.assertIn('rgb_array', self.env.metadata['render_modes'])
    
    def test_metadata_has_name(self):
        """Test that metadata contains name."""
        self.assertIn('name', self.env.metadata)
        self.assertEqual(self.env.metadata['name'], 'overcooked_marl_v0')
    
    def test_reset_signature(self):
        """Test that reset() has correct signature."""
        import inspect
        sig = inspect.signature(self.env.reset)
        params = list(sig.parameters.keys())
        
        # Should have seed and options parameters
        self.assertIn('seed', params)
        self.assertIn('options', params)
    
    def test_reset_returns_observations_and_infos(self):
        """Test that reset() returns (observations, infos)."""
        result = self.env.reset()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        observations, infos = result
        self.assertIsInstance(observations, dict)
        self.assertIsInstance(infos, dict)
    
    def test_reset_with_seed(self):
        """Test that reset() accepts seed parameter."""
        try:
            obs, info = self.env.reset(seed=42)
            self.assertIsInstance(obs, dict)
            self.assertIsInstance(info, dict)
        except Exception as e:
            self.fail(f"reset(seed=42) raised {type(e).__name__}: {e}")
    
    def test_reset_with_options(self):
        """Test that reset() accepts options parameter."""
        try:
            obs, info = self.env.reset(options={})
            self.assertIsInstance(obs, dict)
            self.assertIsInstance(info, dict)
        except Exception as e:
            self.fail(f"reset(options={{}}) raised {type(e).__name__}: {e}")
    
    def test_step_signature(self):
        """Test that step() has correct signature."""
        import inspect
        sig = inspect.signature(self.env.step)
        params = list(sig.parameters.keys())
        
        # Should have actions parameter
        self.assertIn('actions', params)
    
    def test_step_returns_five_values(self):
        """Test that step() returns 5 values (obs, rew, term, trunc, info)."""
        self.env.reset()
        actions = {agent: 0 for agent in self.env.agents}
        result = self.env.step(actions)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        
        observations, rewards, terminations, truncations, infos = result
        self.assertIsInstance(observations, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminations, dict)
        self.assertIsInstance(truncations, dict)
        self.assertIsInstance(infos, dict)
    
    def test_observation_space_method(self):
        """Test that observation_space() method exists."""
        self.assertTrue(hasattr(self.env, 'observation_space'))
        for agent in self.env.possible_agents:
            space = self.env.observation_space(agent)
            self.assertIsNotNone(space)
    
    def test_action_space_method(self):
        """Test that action_space() method exists."""
        self.assertTrue(hasattr(self.env, 'action_space'))
        for agent in self.env.possible_agents:
            space = self.env.action_space(agent)
            self.assertIsNotNone(space)
    
    def test_render_method(self):
        """Test that render() method exists."""
        self.assertTrue(hasattr(self.env, 'render'))
        result = self.env.render()
        self.assertIsNone(result)  # render_mode=None
    
    def test_close_method(self):
        """Test that close() method exists."""
        self.assertTrue(hasattr(self.env, 'close'))
        try:
            self.env.close()
        except Exception as e:
            self.fail(f"close() raised {type(e).__name__}: {e}")
    
    def test_observe_method(self):
        """Test that observe() method exists (PettingZoo-specific)."""
        self.assertTrue(hasattr(self.env, 'observe'))
        self.env.reset()
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            self.assertIsInstance(obs, np.ndarray)
    
    def test_state_method(self):
        """Test that get_state() method exists (PettingZoo-specific)."""
        self.assertTrue(hasattr(self.env, 'get_state'))
        state = self.env.get_state()
        self.assertIsInstance(state, dict)
    
    def test_possible_agents_attribute(self):
        """Test that possible_agents attribute exists."""
        self.assertTrue(hasattr(self.env, 'possible_agents'))
        self.assertIsInstance(self.env.possible_agents, list)
        self.assertGreater(len(self.env.possible_agents), 0)
    
    def test_agents_attribute(self):
        """Test that agents attribute exists."""
        self.assertTrue(hasattr(self.env, 'agents'))
        self.assertIsInstance(self.env.agents, list)
    
    def test_agents_reset_behavior(self):
        """Test that agents list is reset properly."""
        self.env.reset()
        self.assertEqual(len(self.env.agents), len(self.env.possible_agents))
    
    def test_full_episode_workflow(self):
        """Test a complete episode following PettingZoo workflow."""
        # Reset
        observations, infos = self.env.reset(seed=123)
        self.assertEqual(len(observations), len(self.env.agents))
        
        # Run a few steps
        for _ in range(10):
            actions = {agent: 0 for agent in self.env.agents}  # All move up
            obs, rewards, terms, truncs, infos = self.env.step(actions)
            
            # Verify all return values are dictionaries
            self.assertIsInstance(obs, dict)
            self.assertIsInstance(rewards, dict)
            self.assertIsInstance(terms, dict)
            self.assertIsInstance(truncs, dict)
            self.assertIsInstance(infos, dict)
            
            # Verify all agents are accounted for
            for agent in self.env.agents:
                self.assertIn(agent, obs)
                self.assertIn(agent, rewards)
                self.assertIn(agent, terms)
                self.assertIn(agent, truncs)
                self.assertIn(agent, infos)
        
        # Close
        self.env.close()
    
    def test_observation_shape_consistency(self):
        """Test that observations match declared observation space."""
        self.env.reset()
        
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            space = self.env.observation_space(agent)
            
            # Check shape matches
            self.assertEqual(obs.shape, space.shape)
            
            # Check dtype matches
            self.assertEqual(obs.dtype, space.dtype)
    
    def test_action_space_contains_actions(self):
        """Test that action space contains valid actions."""
        for agent in self.env.possible_agents:
            action_space = self.env.action_space(agent)
            
            # For Discrete space, should have n actions
            self.assertTrue(hasattr(action_space, 'n'))
            self.assertEqual(action_space.n, 5)  # 4 movements + interact
    
    def test_render_modes(self):
        """Test different render modes."""
        # Test human mode
        env_human = OvercookedMARL(render_mode="human", verbose=False)
        env_human.reset()
        result = env_human.render()
        # human mode should print and return None
        self.assertIsNone(result)
        env_human.close()
        
        # Test rgb_array mode
        env_rgb = OvercookedMARL(render_mode="rgb_array", verbose=False)
        env_rgb.reset()
        result = env_rgb.render()
        # rgb_array mode should return numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 3)  # Should be H x W x C
        env_rgb.close()


class TestPettingZooMethodOrder(unittest.TestCase):
    """Test that methods follow PettingZoo conventions."""
    
    def test_required_methods_exist(self):
        """Test that all required PettingZoo methods exist."""
        from overcooked_marl_env.env import OvercookedMARL
        
        required_methods = ['reset', 'step', 'render', 'observation_space', 'action_space', 'observe', 'get_state', 'close']
        
        for method in required_methods:
            self.assertTrue(hasattr(OvercookedMARL, method),
                          f"Method '{method}' not found in OvercookedMARL")


if __name__ == '__main__':
    unittest.main()
