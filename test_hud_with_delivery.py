"""
Test HUD rendering with actual deliveries to verify order tracking.
"""
from overcooked_marl_env import OvercookedMARL
import time

def test_hud_with_delivery():
    """Test HUD showing deliveries being removed from orders."""
    print("Testing HUD with deliveries...")
    
    # Use default map which is smaller and easier to test
    env = OvercookedMARL(
        render_mode="human",
        map_path="maps/default_map.txt",
        verbose=True
    )
    
    obs, info = env.reset()
    print("\nEnvironment reset successfully!\n")
    env.render()
    time.sleep(1)
    
    # Execute 100 steps with random actions to try to get some deliveries
    print("\nExecuting 100 random steps...\n")
    for step in range(100):
        actions = {
            agent: env.action_space(agent).sample() 
            for agent in env.agents
        }
        
        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        
        # Check for deliveries
        delivered_count = len(env.state.get("delivered_recipes", []))
        if delivered_count > 0:
            print(f"\n{'='*60}")
            print(f"DELIVERIES DETECTED! Total delivered: {delivered_count}")
            print(f"{'='*60}\n")
            time.sleep(2)  # Pause to see the change
        else:
            time.sleep(0.1)
        
        # Check if done
        if all(terminations.values()) or all(truncations.values()):
            print("\nEpisode finished!")
            break
    
    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_hud_with_delivery()
