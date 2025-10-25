"""
Cooperation Test: Two agents make a salad together.

SIMPLIFIED Strategy:
1. Agent 0: Process tomato completely
2. Agent 1: Process lettuce completely  
3. Agent 0: Pick plate and add tomato
4. Agent 0: Place plate+tomato on counter
5. Agent 1: Pick plate+tomato, add lettuce
6. Agent 1: Deliver the complete salad

Map default_map.txt (5x5):
  0   1   2   3   4
0 #   C   #   P   #
1 #       0       #
2 #               D
3 #       1       #
4 #   I   I   T   #

Important positions:
- Chopping Board (C): (1,0)
- Plate Dispenser (P): (3,0)
- Delivery (D): (4,2)
- Tomato Dispenser (I): (1,4)
- Lettuce Dispenser (I): (2,4)
- Trash/Counter (T): (3,4)
"""

import time
from overcooked_marl_env.env import OvercookedMARL


def test_cooperative_salad():
    """Test where two agents cooperate to make a salad."""
    print("\n" + "="*60)
    print("TEST: Cooperation to Make Salad")
    print("="*60)
    print("\nStrategy:")
    print("  1. Agent 0: Process tomato")
    print("  2. Agent 1: Process lettuce")
    print("  3. Agent 0: Pick plate + add tomato + place on counter")
    print("  4. Agent 1: Pick plate+tomato + add lettuce")
    print("  5. Agent 1: Deliver salad")
    print("="*60 + "\n")
    
    env = OvercookedMARL(render_mode="human", verbose=False)
    obs, info = env.reset()
    
    # Agent 0 starts at (2,1)
    # Agent 1 starts at (2,3)
    
    actions = [
        # ===== PHASE 1: Agent 0 picks and processes tomato =====
        ("Agent 0: Go left", {
            "agent_0": 2,  # LEFT: (2,1) -> (1,1)
            "agent_1": 4
        }),
        ("Agent 0: Go down", {
            "agent_0": 1,  # DOWN: (1,1) -> (1,2)
            "agent_1": 4
        }),
        ("Agent 0: Go down", {
            "agent_0": 1,  # DOWN: (1,2) -> (1,3)
            "agent_1": 4
        }),
        ("Agent 0: Turn down (tomato dispenser)", {
            "agent_0": 1,  # DOWN: collide, turn South, point to (1,4)
            "agent_1": 4
        }),
        ("Agent 0: Pick tomato", {
            "agent_0": 4,  # INTERACT: pick tomato from (1,4)
            "agent_1": 4
        }),
        ("Agent 0: Go up", {
            "agent_0": 0,  # UP: (1,3) -> (1,2)
            "agent_1": 4
        }),
        ("Agent 0: Go up (adjacent to chopping)", {
            "agent_0": 0,  # UP: (1,2) -> (1,1)
            "agent_1": 4
        }),
        ("Agent 0: Turn up (point to chopping)", {
            "agent_0": 0,  # UP: collide, turn North, point to (1,0)
            "agent_1": 4
        }),
        ("Agent 0: Cut tomato 1/3", {
            "agent_0": 4,  # INTERACT: process at (1,0)
            "agent_1": 4
        }),
        ("Agent 0: Cut tomato 2/3", {
            "agent_0": 4,  # INTERACT: process at (1,0)
            "agent_1": 4
        }),
        ("Agent 0: Cut tomato 3/3 - READY!", {
            "agent_0": 4,  # INTERACT: process at (1,0)
            "agent_1": 4
        }),
        
        # ===== PHASE 2: Agent 1 picks and processes lettuce =====
        ("Agent 1: Turn down (lettuce dispenser)", {
            "agent_0": 4,
            "agent_1": 1   # DOWN: collide, turn South, point to (2,4)
        }),
        ("Agent 1: Pick lettuce", {
            "agent_0": 4,
            "agent_1": 4   # INTERACT: pick lettuce from (2,4)
        }),
        ("Agent 1: Go up", {
            "agent_0": 4,
            "agent_1": 0   # UP: (2,3) -> (2,2)
        }),
        ("Agent 1: Go left", {
            "agent_0": 4,
            "agent_1": 2   # LEFT: (2,2) -> (1,2)
        }),
        ("Agent 1: Go up (adjacent to chopping)", {
            "agent_0": 4,
            "agent_1": 0   # UP: (1,2) -> (1,1)
        }),
        
        # Agent 0 needs to move out of the way!
        ("Agent 0: Move right", {
            "agent_0": 3,  # RIGHT: (1,1) -> (2,1)
            "agent_1": 4
        }),
        
        ("Agent 1: Turn up (point to chopping)", {
            "agent_0": 4,
            "agent_1": 0   # UP: collide, turn North, point to (1,0)
        }),
        ("Agent 1: Cut lettuce 1/3", {
            "agent_0": 4,
            "agent_1": 4   # INTERACT: process at (1,0)
        }),
        ("Agent 1: Cut lettuce 2/3", {
            "agent_0": 4,
            "agent_1": 4   # INTERACT: process at (1,0)
        }),
        ("Agent 1: Cut lettuce 3/3 - READY!", {
            "agent_0": 4,
            "agent_1": 4   # INTERACT: process at (1,0)
        }),
        
        # ===== PHASE 3: Agent 0 picks plate and adds tomato =====
        ("Agent 0: Go right", {
            "agent_0": 3,  # RIGHT: (2,1) -> (3,1)
            "agent_1": 4
        }),
        ("Agent 0: Turn up (plate dispenser)", {
            "agent_0": 0,  # UP: collide, turn North, point to (3,0)
            "agent_1": 4
        }),
        ("Agent 0: Pick plate (automatically combines with tomato)", {
            "agent_0": 4,  # INTERACT: pick plate from (3,0), combine with tomato
            "agent_1": 4
        }),
        
        # ===== PHASE 4: Agent 0 places plate+tomato on counter T =====
        ("Agent 0: Go down", {
            "agent_0": 1,  # DOWN: (3,1) -> (3,2)
            "agent_1": 4
        }),
        ("Agent 0: Go down", {
            "agent_0": 1,  # DOWN: (3,2) -> (3,3)
            "agent_1": 4
        }),
        ("Agent 0: Turn down (counter T)", {
            "agent_0": 1,  # DOWN: collide, turn South, point to (3,4)
            "agent_1": 4
        }),
        ("Agent 0: Place plate+tomato on counter", {
            "agent_0": 4,  # INTERACT: place plate on (3,4)
            "agent_1": 4
        }),
        
        # Agent 0 needs to MOVE AWAY so it doesn't pick up the plate again!
        ("Agent 0: Go up to move away", {
            "agent_0": 0,  # UP: (3,3) -> (3,2)
            "agent_1": 4
        }),
        
        # ===== PHASE 5: Agent 1 goes to counter and picks plate+tomato =====
        ("Agent 1: Exit chopping", {
            "agent_0": 0,  # UP: Agent 0 continues going up
            "agent_1": 1   # DOWN: (1,1) -> (1,2)
        }),
        ("Agent 1: Go right", {
            "agent_0": 0,  # UP: Agent 0 continues moving away
            "agent_1": 3   # RIGHT: (1,2) -> (2,2)
        }),
        ("Agent 1: Go right", {
            "agent_0": 0,  # UP: Agent 0 goes to (3,1)
            "agent_1": 3   # RIGHT: (2,2) -> (3,2)
        }),
        ("Agent 1: Go down (adjacent to counter)", {
            "agent_0": 2,  # LEFT: Agent 0 goes to (2,1) - away from counter!
            "agent_1": 1   # DOWN: (3,2) -> (3,3)
        }),
        ("Agent 1: Turn down (point to counter T)", {
            "agent_0": 2,  # LEFT: Agent 0 continues moving away
            "agent_1": 1   # DOWN: collide, turn South, point to (3,4)
        }),
        ("Agent 1: Pick plate+tomato and combine with lettuce!", {
            "agent_0": 2,  # LEFT: Agent 0 goes to (1,1)
            "agent_1": 4   # INTERACT: pick plate from (3,4), combine with lettuce!
        }),
        
        # ===== PHASE 6: Agent 1 delivers the salad =====
        ("Agent 1: Go up", {
            "agent_0": 2,  # LEFT: Agent 0 stays still
            "agent_1": 0   # UP: (3,3) -> (3,2)
        }),
        ("Agent 1: Turn right (delivery)", {
            "agent_0": 2,  # LEFT: Agent 0 stays still
            "agent_1": 3   # RIGHT: collide, turn East, point to (4,2)
        }),
        ("Agent 1: DELIVER COMPLETE SALAD!", {
            "agent_0": 2,  # LEFT: Agent 0 stays still
            "agent_1": 4   # INTERACT: deliver at (4,2)!
        }),
    ]
    
    step_count = 0
    for desc, action_dict in actions:
        step_count += 1
        print(f"\n{'='*60}")
        print(f"STEP {step_count}: {desc}")
        print('='*60)
        
        obs, rewards, dones, truncs, info = env.step(action_dict)
        env.render()
        
        # Show agent information
        for agent_id in ["agent_0", "agent_1"]:
            inv = info[agent_id]['inventory']
            rew = rewards[agent_id]
            print(f"{agent_id}: Inventory = {inv:30s} | Reward = {rew:+.3f}")
        
        time.sleep(0.05)
        
        # Check if completed
        if info['agent_0']['dishes_remaining'] < 3:
            print("\n" + "="*60)
            print("*** SALAD DELIVERED SUCCESSFULLY! ***")
            print("="*60)
            break
    
    print("\n" + "="*60)
    print(f"Test completed in {step_count} steps!")
    print(f"Delivered recipes: {info['agent_0']['delivered_recipes']}")
    print(f"Remaining dishes: {info['agent_0']['dishes_remaining']}")
    
    # Check success
    if len(info['agent_0']['delivered_recipes']) > 0:
        print("\n[OK] TEST PASSED! Salad delivered successfully!")
        return True
    else:
        print("\n[FAILED] No recipe was delivered.")
        return False
    
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" COOPERATION TEST - TEAM SALAD")
    print("="*60)
    
    try:
        success = test_cooperative_salad()
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            import pygame
            pygame.quit()
        except:
            pass
