"""
Test script to verify that env.py works with the new MapLoader system.
"""

from overcooked_marl_env.env import OvercookedMARL

def test_env_initialization():
    """Test that environment initializes correctly with new map system."""
    print("=" * 60)
    print("Test 1: Environment Initialization with Default Map")
    print("=" * 60)
    
    env = OvercookedMARL(
        render_mode=None,
        map_path="maps/default_map.txt",
        num_fixed_agents=2,
        verbose=True
    )
    
    print(f"\n✓ Environment created successfully")
    print(f"  - Map size: {env.map_size}")
    print(f"  - Agents: {env.agents}")
    print(f"  - Recipe order: {env.recipe_order}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Environment reset successfully")
    print(f"  - Observation keys: {list(obs.keys())}")
    print(f"  - Info keys: {list(info.keys())}")
    
    # Test step
    actions = {agent: 0 for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print(f"\n✓ Environment step successful")
    print(f"  - Rewards: {rewards}")
    print(f"  - Terminations: {terminations}")
    
    env.close()
    print(f"\n✓ Environment closed successfully")
    
    return True

def test_tutorial_map():
    """Test loading the tutorial map."""
    print("\n" + "=" * 60)
    print("Test 2: Loading Tutorial Map")
    print("=" * 60)
    
    env = OvercookedMARL(
        render_mode=None,
        map_path="overcooked_marl_env/map/maps/tutorial.txt",
        num_fixed_agents=2,
        verbose=True
    )
    
    print(f"\n✓ Tutorial map loaded successfully")
    print(f"  - Map size: {env.map_size}")
    print(f"  - Agents: {env.agents}")
    print(f"  - Recipe order: {env.recipe_order}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Tutorial environment reset successfully")
    
    # Get agent positions from state
    agent_positions = env.state["agent_pos"]
    print(f"  - Agent positions: {agent_positions}")
    
    # Check info
    for agent_id, agent_info in info.items():
        print(f"  - {agent_id}: {agent_info['next_expected_recipe']}")
    
    env.close()
    print(f"\n✓ Tutorial environment closed successfully")
    
    return True

def test_multiple_resets():
    """Test multiple resets to ensure consistency."""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Resets")
    print("=" * 60)
    
    env = OvercookedMARL(
        render_mode=None,
        map_path="maps/default_map.txt",
        num_fixed_agents=2,
        verbose=False  # Less verbose for this test
    )
    
    for i in range(3):
        obs, info = env.reset()
        print(f"\n✓ Reset {i+1} successful")
        print(f"  - Map size: {env.map_size}")
        print(f"  - Recipe order: {env.recipe_order}")
    
    env.close()
    print(f"\n✓ Multiple resets completed successfully")
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing OvercookedMARL with New MapLoader System")
    print("=" * 60)
    
    try:
        test_env_initialization()
        test_tutorial_map()
        test_multiple_resets()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe new MapLoader system is compatible with env.py!")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
