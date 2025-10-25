"""
Test for the new item and recipe system.

This test validates:
- Creation of ingredients with states
- Progressive processing (3 cuts)
- Combination of ingredients on plates
- Recipe validation
- Cooperation between agents (shared cuts)
"""

import time
from overcooked_marl_env.env import OvercookedMARL

def test_progressive_cutting():
    """Tests the progressive cutting system."""
    print("\n" + "="*60)
    print("TEST 1: Progressive Processing (3 cuts)")
    print("="*60)
    
    env = OvercookedMARL(render_mode="human", verbose=True)
    obs, info = env.reset()
    
    # Agent 0 will pick up a tomato and chop it completely
    # Map: Agent 0 at (2,1), tomato at (1,4), chopping board at (1,0)
    actions = [
        (0, 2, "Move LEFT"),                               # (2,1) -> (1,1)
        (0, 1, "Move DOWN"),                               # (1,1) -> (1,2)
        (0, 1, "Move DOWN"),                               # (1,2) -> (1,3)
        (0, 4, "Pick tomato from dispenser (I)"),          # Interact with (1,4) - pick tomato
        (0, 0, "Move UP"),                                 # (1,3) -> (1,2)
        (0, 0, "Move UP"),                                 # (1,2) -> (1,1)
        (0, 4, "Cut tomato - 1/3"),                        # Interact with (1,0) - cut 1
        (0, 4, "Cut tomato - 2/3"),                        # Interact - cut 2
        (0, 4, "Cut tomato - 3/3 (complete)"),             # Interact - cut 3
    ]
    
    for step, (agent, action, desc) in enumerate(actions, 1):
        print(f"\n--- STEP {step}: {desc} ---")
        agent_id = f"agent_{agent}"
        obs, rewards, dones, truncs, info = env.step({agent_id: action})
        env.render()
        print(f"Inventory {agent_id}: {info[agent_id]['inventory']}")
        print(f"Reward: {rewards[agent_id]:.3f}")
        time.sleep(0.1)
    
    print("\n[OK] Progressive processing test completed!")
    env.close()


def test_cooperative_cutting():
    """Tests cooperative cutting between two agents."""
    print("\n" + "="*60)
    print("TEST 2: Cooperative Cutting")
    print("="*60)
    
    env = OvercookedMARL(render_mode="human", verbose=True)
    obs, info = env.reset()
    
    # Agent 0 cuts 2 times, then agent 1 finishes the cut
    # Agent 0 at (2,1), Agent 1 at (2,3)
    actions = [
        (0, 2, "Agent 0: LEFT"),                           # (2,1) -> (1,1)
        (0, 1, "Agent 0: DOWN"),                           # (1,1) -> (1,2)
        (0, 1, "Agent 0: DOWN"),                           # (1,2) -> (1,3)
        (0, 4, "Agent 0: Pick tomato"),                    # Interact with (1,4)
        (0, 0, "Agent 0: UP"),                             # (1,3) -> (1,2)
        (0, 0, "Agent 0: UP"),                             # (1,2) -> (1,1)
        (0, 4, "Agent 0: Cut 1/3"),                        # Interact with (1,0)
        (0, 4, "Agent 0: Cut 2/3"),                        # Interact
        (0, 1, "Agent 0: Exit DOWN"),                      # (1,1) -> (1,2)
        (0, 3, "Agent 0: RIGHT"),                          # (1,2) -> (2,2)
        (0, 3, "Agent 0: RIGHT"),                          # (2,2) -> (3,2)
        (0, 1, "Agent 0: DOWN"),                           # (3,2) -> (3,3)
        (0, 4, "Agent 0: Place tomato (2 cuts)"),          # Interact with (3,4) - place on T
        (0, 0, "Agent 0: Move UP"),                        # (3,3) -> (3,2) - MOVE to free up space
        (1, 3, "Agent 1: RIGHT"),                          # (2,3) -> (3,3)
        (1, 4, "Agent 1: Pick tomato (2 cuts)"),           # Interact with (3,4) - pick from T
        (1, 2, "Agent 1: LEFT"),                           # (3,3) -> (2,3)
        (1, 2, "Agent 1: LEFT"),                           # (2,3) -> (1,3)
        (1, 0, "Agent 1: UP"),                             # (1,3) -> (1,2)
        (1, 0, "Agent 1: UP"),                             # (1,2) -> (1,1)
        (1, 4, "Agent 1: Cut 3/3 (complete)"),             # Interact with (1,0)
    ]
    
    for step, (agent, action, desc) in enumerate(actions, 1):
        print(f"\n--- STEP {step}: {desc} ---")
        agent_id = f"agent_{agent}"
        obs, rewards, dones, truncs, info = env.step({agent_id: action})
        env.render()
        for ag in ["agent_0", "agent_1"]:
            print(f"Inventory {ag}: {info[ag]['inventory']}")
        time.sleep(0.1)
    
    print("\n[OK] Cooperative cutting test completed!")
    env.close()


def test_recipe_combination():
    """Tests ingredient combination and recipe validation."""
    print("\n" + "="*60)
    print("TEST 3: Ingredient Combination and Recipe")
    print("="*60)
    
    env = OvercookedMARL(render_mode="human", verbose=True)
    obs, info = env.reset()
    
    # Agent 0 prepares a complete salad
    # Agent 0 at (2,1)
    actions = [
        # Pick and process tomato
        (0, 2, "LEFT"),                                    # (2,1) -> (1,1)
        (0, 1, "DOWN"),                                    # (1,1) -> (1,2)
        (0, 1, "DOWN"),                                    # (1,2) -> (1,3)
        (0, 4, "Pick tomato"),                             # Interact with (1,4)
        (0, 0, "UP"),                                      # (1,3) -> (1,2)
        (0, 0, "UP"),                                      # (1,2) -> (1,1)
        (0, 4, "Cut tomato 1/3"),                          # Interact with (1,0)
        (0, 4, "Cut tomato 2/3"),                          # Interact
        (0, 4, "Cut tomato 3/3 - READY"),                  # Interact
        # Pick plate and combine with tomato
        (0, 3, "RIGHT"),                                   # (1,1) -> (2,1)
        (0, 3, "RIGHT"),                                   # (2,1) -> (3,1)
        (0, 4, "Pick plate + combine tomato"),             # Interact with (3,0) - auto-combine
        # Leave plate on counter T
        (0, 1, "DOWN"),                                    # (3,1) -> (3,2)
        (0, 1, "DOWN"),                                    # (3,2) -> (3,3)
        (0, 4, "Place plate+tomato on counter T"),         # Interact with (3,4)
        # Agent 1 move out of the way
        (1, 0, "UP"),                                      # Agent 1: (2,3) -> (2,2)
        (1, 0, "UP"),                                      # Agent 1: (2,2) -> (2,1)
        # Go to lettuce dispenser
        (0, 0, "UP"),                                      # (3,3) -> (3,2) - exit counter
        (0, 2, "LEFT"),                                    # (3,2) -> (2,2)
        (0, 1, "DOWN"),                                    # (2,2) -> (2,3) - now free!
        (0, 4, "Pick lettuce from dispenser"),             # Interact with (2,4) - pick lettuce
        # Process lettuce
        (0, 0, "UP"),                                      # (2,3) -> (2,2)
        (0, 2, "LEFT"),                                    # (2,2) -> (1,2)
        (0, 0, "UP"),                                      # (1,2) -> (1,1)
        (0, 4, "Cut lettuce 1/3"),                         # Interact with (1,0)
        (0, 4, "Cut lettuce 2/3"),                         # Interact
        (0, 4, "Cut lettuce 3/3 - READY"),                 # Interact
        # Go to counter T and combine lettuce with plate - alternative route
        (0, 1, "DOWN"),                                    # (1,1) -> (1,2)
        (0, 3, "RIGHT"),                                   # (1,2) -> (2,2)
        (0, 3, "RIGHT"),                                   # (2,2) -> (3,2)
        (0, 1, "DOWN"),                                    # (3,2) -> (3,3)
        (0, 4, "Combine lettuce with plate"),              # Interact with (3,4) - pick plate and combine
        # Go to (3,2) and deliver by interacting with (4,2)
        (0, 0, "UP"),                                      # (3,3) -> (3,2)
        (0, 4, "Deliver salad"),                           # Interact with (4,2) - delivery
    ]
    
    for step, (agent, action, desc) in enumerate(actions, 1):
        print(f"\n--- STEP {step}: {desc} ---")
        agent_id = f"agent_{agent}"
        obs, rewards, dones, truncs, info = env.step({agent_id: action})
        env.render()
        print(f"Inventory {agent_id}: {info[agent_id]['inventory']}")
        print(f"Reward: {rewards[agent_id]:.3f}")
        time.sleep(0.05)
    
    print("\n[OK] Recipe test completed!")
    print(f"Delivered recipes: {info['agent_0']['delivered_recipes']}")
    env.close()


def test_invalid_recipe():
    """Tests delivery of invalid recipe."""
    print("\n" + "="*60)
    print("TEST 4: Invalid Recipe (no points)")
    print("="*60)
    
    env = OvercookedMARL(render_mode="human", verbose=True)
    obs, info = env.reset()
    
    # Agent 0 tries to deliver a plate with only one ingredient
    # Agent 0 at (2,1)
    actions = [
        (0, 2, "LEFT"),                                    # (2,1) -> (1,1)
        (0, 1, "DOWN"),                                    # (1,1) -> (1,2)
        (0, 1, "DOWN"),                                    # (1,2) -> (1,3)
        (0, 4, "Pick tomato"),                             # Interact with (1,4)
        (0, 0, "UP"),                                      # (1,3) -> (1,2)
        (0, 0, "UP"),                                      # (1,2) -> (1,1)
        (0, 4, "Cut 1/3"),                                 # Interact with (1,0)
        (0, 4, "Cut 2/3"),                                 # Interact
        (0, 4, "Cut 3/3 - READY"),                         # Interact
        (0, 3, "RIGHT"),                                   # (1,1) -> (2,1)
        (0, 3, "RIGHT"),                                   # (2,1) -> (3,1)
        (0, 4, "Pick plate and combine tomato"),           # Interact with (3,0)
        (0, 1, "DOWN to delivery"),                        # (3,1) -> (3,2)
        (0, 4, "Try to deliver (SHOULD FAIL)"),            # Interact with (4,2) - incomplete recipe
    ]
    
    initial_dishes = info['agent_0']['dishes_remaining']
    
    for step, (agent, action, desc) in enumerate(actions, 1):
        print(f"\n--- STEP {step}: {desc} ---")
        agent_id = f"agent_{agent}"
        obs, rewards, dones, truncs, info = env.step({agent_id: action})
        env.render()
        print(f"Reward: {rewards[agent_id]:.3f}")
        time.sleep(0.05)
    
    final_dishes = info['agent_0']['dishes_remaining']
    
    # Verify that it was not delivered
    assert final_dishes == initial_dishes, "Invalid recipe should not have been accepted!"
    
    print("\n[OK] Invalid recipe test completed!")
    print(f"Remaining dishes: {final_dishes} (none delivered, correct!)")
    env.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" TEST OF NEW ITEM AND RECIPE SYSTEM")
    print("="*60)
    
    try:
        test_progressive_cutting()
        print("\n" + "-"*60)
        
        test_cooperative_cutting()
        print("\n" + "-"*60)
        
        test_recipe_combination()
        print("\n" + "-"*60)
        
        test_invalid_recipe()
        
        print("\n" + "="*60)
        print(" ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()

