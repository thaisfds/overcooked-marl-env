import gymnasium as gym
import numpy as np

# PettingZoo imports
try:
    from pettingzoo.utils.conversions import parallel_to_aec_wrapper
    from pettingzoo.test import parallel_api_test
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    print("Warning: PettingZoo not found. Some tests will be skipped.")

# Import the direct class for instantiation
from overcooked_marl_env.env import OvercookedMARL

def action_to_str(action):
    """Convert numeric action to descriptive string"""
    action_map = {
        0: "(Up)",
        1: "(Down)",
        2: "(Left)",
        3: "(Right)",
        4: "(Interact)"
    }
    # Handle numpy int types by converting to Python int
    if hasattr(action, 'item'):
        action = action.item()
    return action_map.get(action, f"Unknown ({action})")

def run_sanity_check(env_class, max_steps=100):
    """
    Runs a basic test on the multi-agent environment to verify if the API is correct.
    """
    
    # 1. Create the Environment
    print("--- 1. Initializing Environment ---")
    
    env = env_class(verbose=False) 
    print(f"Environment created via direct class: {env.metadata['name']}")
        
    print(f"Active Agents: {env.agents}")
    print(f"Action Space for 'agent_0': {env.action_space('agent_0')}")
    print(f"Observation Space for 'agent_0': {env.observation_space('agent_0').shape}")
    print("-" * 30)

    # 2. Reset the Environment
    print("--- 2. Reset (Episode Start) ---")
    observations, info = env.reset(seed=42)
    print(f"Initial Observations: {len(observations)} agents")
    print(f"Initial Info: {info.keys()}")
    
    # Display recipe information if available
    if "recipe_order" in info["agent_0"]:
        print(f"Expected Recipe Order: {info['agent_0']['recipe_order']}")
        print(f"Current Expected Recipe: {info['agent_0']['next_expected_recipe']}")
    print("-" * 30)
    
    total_reward = {agent: 0 for agent in env.agents}
    
    # 3. Execute Simulation Steps
    print(f"--- 3. Running {max_steps} Random Steps ---")
    for step in range(max_steps):
        # 3.1 Generate Random Actions for EACH active agent
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        
        # 3.2 Execute the Step
        # The return is always 5 dictionaries: obs, reward, terminated, truncated, info
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 3.3 Process the Result
        
        # Check if the episode is over for any agent
        done = any(terminations.values()) or any(truncations.values())

        # Accumulate rewards
        for agent, reward in rewards.items():
            total_reward[agent] += reward
        
        if step % 10 == 0 or done:
            print(f"\n[Step {step}]")
            
            # Format actions with more meaningful names
            readable_actions = {}
            for agent, action in actions.items():
                readable_actions[agent] = action_to_str(action)
            print(f"  Actions: {readable_actions}")
            
            print(f"  Rewards: {rewards}")
            print(f"  Total Accumulated: {total_reward}")
            print(f"  Terminated: {terminations}")
            print(f"  Truncated: {truncations}")
        
        if done:
            print("\n!!! EPISODE TERMINATED/TRUNCATED !!!")
            # Show information about delivered recipes
            if infos and "delivered_recipes" in infos["agent_0"]:
                print(f"Delivered Recipes: {infos['agent_0']['delivered_recipes']}")
            break

    # 4. Close the Environment
    env.close()
    print("-" * 30)
    print("Test completed successfully (Structure verified).")

if __name__ == "__main__":
    # PettingZoo environments are typically created directly, not through gym.make
    print("Creating PettingZoo ParallelEnv directly...")
    
    # Create the environment directly (recommended approach for PettingZoo)
    run_sanity_check(OvercookedMARL)
    
    # Run PettingZoo API test if PettingZoo is available
    if HAS_PETTINGZOO:
        print("\n--- Running PettingZoo API test ---")
        env = OvercookedMARL()
        try:
            parallel_api_test(env, num_cycles=10)
            print("PettingZoo API test passed!")
        except Exception as e:
            print(f"PettingZoo API test failed: {e}")
        finally:
            env.close()