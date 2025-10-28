"""
Test script to verify HUD rendering in RGB mode.
"""
import time
from overcooked_marl_env import OvercookedMARL

def test_hud_rendering():
    """Test the new HUD rendering feature."""
    print("Testing HUD rendering...")
    
    # Create environment with RGB rendering
    env = OvercookedMARL(
        render_mode="human",  # This will open a pygame window
        map_path="maps/tutorial.txt"
    )
    
    # Reset environment
    observations, info = env.reset(seed=42)
    print("Environment reset successfully!")
    
    # Run a few steps to accumulate some rewards
    for step in range(20):
        # Random actions for testing
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render the environment (HUD should be visible)
        env.render()
        
        # Print step info
        print(f"\nStep {step + 1}:")
        for agent in env.agents:
            print(f"  {agent}: Reward = {rewards.get(agent, 0):+.3f}")
        
        time.sleep(0.2)  # Slow down for visual inspection
        
        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()):
            print("\nEpisode finished!")
            break
    
    env.close()
    print("\nHUD rendering test completed!")

if __name__ == "__main__":
    test_hud_rendering()
