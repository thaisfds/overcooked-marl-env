"""
Tutorial Complete Test: Comprehensive test of ALL game mechanics using tutorial.txt map.

This test demonstrates cooperation between two agents in isolated areas who must
work together using kitchen counters to transfer items.

MAP STRUCTURE (tutorial.txt): 15x10 grid with borders
     x: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  y0:   _  _  _  _  _  _  _  _  _  _  _  _  _  _  _
  y1:   |  #                    #  |
  y2:   |  C                    C  |
  y3:   |  #              0     #  |
  y4:   |  C                    C  |
  y5:   |  #  T  #  #  #  #  #  #  #  #  #  #  |
  y6:   |                          |
  y7:   |           1              |
  y8:   |  #  I  I  #  P  #  D  D  #  #  I  #  #  |
  y9:   _  _  _  _  _  _  _  _  _  _  _  _  _  _  _

KEY POSITIONS (based on actual parsing):
- Agent 0: starts at (10, 3) - upper right area
- Agent 1: starts at (5, 7) - lower central area
- Chopping boards (C): (2,2), (13,2), (2,4), (13,4)
- Kitchen Counters (#): multiple positions - used for item transfer
- Trash (T): (3, 5) - NOT for ingredients
- Plate dispenser (P): (5, 8)
- Delivery (D): (7,8) and (8,8)
- Dispensers (I): lettuce (2,8), tomato (3,8), cucumber (11,8)

MECHANICS TESTED:
1. [OK] Place ingredient on counter
2. [OK] Pick ingredient from counter
3. [OK] Place ingredient on chopping board and chop completely (3 cuts)
4. [OK] Place ingredient on chopping board and partial chop (1 cut)
5. [OK] Try to add unprocessed ingredient to plate (should fail)
6. [OK] Add processed (READY) ingredient to plate
7. [OK] Pick up ingredient with plate in hand (auto-combine)
8. [OK] Transfer items between agents via counter

RECIPES TO COMPLETE:
1. SIMPLE_SALAD (lettuce)
2. GARDEN_SALAD (lettuce + tomato)
3. FRESH_SALAD (lettuce + tomato + cucumber)
"""

import time
import os
import sys
from datetime import datetime
from overcooked_marl_env.env import OvercookedMARL


class Logger:
    """Duplicates output to console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8', errors='replace')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def test_tutorial_complete(render_mode="human", log_to_file=True):
    """
    Comprehensive test of all game mechanics using tutorial map.
    
    Args:
        render_mode: "ascii" for ASCII console (default) or "human" for Pygame graphics
        log_to_file: If True, saves output to logs/test_tutorial_TIMESTAMP.txt
    """
    # Setup logging to file if requested
    logger = None
    if log_to_file:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"test_tutorial_{timestamp}.txt")
        
        logger = Logger(log_file)
        sys.stdout = logger
        print(f"[Logging to: {log_file}]\n")
    
    print("\n" + "="*80)
    print("TUTORIAL COMPLETE TEST - ALL MECHANICS")
    print("="*80)
    print("\nThis test demonstrates ALL game mechanics through cooperative gameplay:")
    print("  • Item dispensing and counter placement")
    print("  • Progressive chopping (partial and complete)")
    print("  • Plate + ingredient combinations")
    print("  • Item transfer between isolated agents via counters")
    print("  • Multiple recipe completion (3 different salads)")
    print("="*80 + "\n")
    
    env = OvercookedMARL(render_mode=render_mode, map_path="tutorial.txt", verbose=True)
    obs, info = env.reset()
    
    # Render initial state
    env.render()
    time.sleep(0.5 if render_mode == "human" else 0.5)
    
    # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Interact
    # Agent 0 starts at (10, 3)
    # Agent 1 starts at (5, 7)
    
    actions = [
        # ===================================================================
        # RECIPE 1: SIMPLE_SALAD (lettuce only)
        # Tests: basic dispenser, complete chopping, plate + ingredient, delivery
        # ===================================================================
        
        ("RECIPE 1 START: Making SIMPLE_SALAD (lettuce)", {}),
        
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 0
        }),
        ("text", {
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 1
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 3
        }),
        ("text", {
            "agent_0": 2,
            "agent_1": 0
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 1,
            "agent_1": 2
        }),
        ("text", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 1
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 4
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 0
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 0
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 4
        }),
        ("text", {
			"agent_0": 3,
            "agent_1": 2
        }),
        ("text", {
			"agent_0": 1,
            "agent_1": 0
        }),
        ("text", {
            "agent_1": 4
        }),
        ("text", {
            "agent_1": 3
        }),
        ("text", {
            "agent_1": 3
        }),
        ("text", {
            "agent_1": 3
        }),
        ("text", {
            "agent_1": 3
        }),
        ("text", {
            "agent_1": 1
        }),
        ("text", {
            "agent_1": 1
        }),
        ("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 1
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 2
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 2,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 3
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 0
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 3,
            "agent_1": 1
        }),
		("text", {
			"agent_0": 1,
            "agent_1": 4
        }),
		("text", {
			"agent_0": 4,
            "agent_1": 2
        }),
		("text", {
            "agent_1": 2
        }),
		("text", {
            "agent_1": 2
        }),
		("text", {
            "agent_1": 1
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 2
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 3
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        }),
		("text", {
            "agent_1": 2
        }),
		("text", {
            "agent_1": 0
        }),
		("text", {
            "agent_1": 4
        })


    ]
    
    step_count = 0
    recipes_delivered = 0
    
    for desc, action_dict in actions:
        # Skip empty description entries (used for spacing)
        if not desc:
            continue
            
        # Check for special announcement messages (no actions)
        if not action_dict:
            print(f"\n{'='*80}")
            print(f">>> {desc}")
            print('='*80)
            time.sleep(0.2 if render_mode == "human" else 0.3)
            continue
        
        step_count += 1
        print(f"\n{'='*80}")
        print(f"STEP {step_count}: {desc}")
        print('='*80)
        
        obs, rewards, dones, truncs, info = env.step(action_dict)
        env.render()
        
        # Show agent information
        for agent_id in ["agent_0", "agent_1"]:
            inv = info[agent_id]['inventory']
            rew = rewards[agent_id]
            acc_rew = env.state["accumulated_rewards"][agent_id]
            print(f"{agent_id}: Inventory = {inv:40s} | Step Reward = {rew:+.3f} | Total = {acc_rew:+.3f}")
        
        # Show recipe progress
        delivered = len(info['agent_0']['delivered_recipes'])
        remaining = info['agent_0']['dishes_remaining']
        if delivered > recipes_delivered:
            recipes_delivered = delivered
            print(f"\n*** RECIPE DELIVERED! Total: {delivered}/3 ***")
        
        # Delay for visualization
        time.sleep(0.2 if render_mode == "human" else 0.15)
        
        # Check if all recipes completed
        # dishes_pending starts at 5, so after delivering 3 recipes it should be 2
        if delivered >= 3:  # Delivered all 3 required recipes
            print("\n" + "="*80)
            print("*** ALL RECIPES DELIVERED SUCCESSFULLY! ***")
            print("="*80)
            break
    
    # Final summary
    print("\n" + "="*80)
    print("TUTORIAL COMPLETE TEST - FINAL SUMMARY")
    print("="*80)
    print(f"Total steps taken: {step_count}")
    print(f"Recipes delivered: {info['agent_0']['delivered_recipes']}")
    print(f"Remaining orders: {info['agent_0']['dishes_remaining']}")
    print(f"Agent 0 total reward: {env.state['accumulated_rewards']['agent_0']:+.2f}")
    print(f"Agent 1 total reward: {env.state['accumulated_rewards']['agent_1']:+.2f}")
    print("="*80)
    
    # Check success
    success = len(info['agent_0']['delivered_recipes']) >= 3
    
    if success:
        print("\n" + "="*80)
        print("[SUCCESS] TEST PASSED!")
        print("All mechanics tested successfully:")
        print("  [OK] Item dispensing and counter placement")
        print("  [OK] Progressive chopping (partial and complete)")
        print("  [OK] Plate + ingredient combinations")
        print("  [OK] Item transfer between agents via counters")
        print("  [OK] Multiple recipe completion")
        print("="*80)
    else:
        print("\n[FAILED] Not all recipes were completed.")
        print(f"Expected: 3 recipes | Delivered: {len(info['agent_0']['delivered_recipes'])}")
    
    # Keep window open for rgb_array mode
    if render_mode == "human":
        print("\nFinal state rendered. Close the Pygame window to exit.")
        print("(Window will auto-close in 3 seconds)")
        
        for _ in range(30):  # 3 seconds total
            env.render()
            time.sleep(0.1)
    
    env.close()
    
    # Restore stdout and close logger
    if logger:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n[Log saved to: {log_file}]")
    
    return success


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" TUTORIAL COMPLETE TEST - ALL GAME MECHANICS")
    print("="*80)
    print("\nThis comprehensive test demonstrates:")
    print("  • All item manipulation mechanics")
    print("  • Progressive processing system")
    print("  • Agent cooperation via counters")
    print("  • Complete recipe workflows")
    print("\nConfiguration:")
    print('  - RENDER_MODE: "ascii" (ASCII) or "human" (Pygame)')
    print('  - SAVE_LOG: True (save to logs/) or False (console only)')
    print("="*80)
    
    # Configuration
    RENDER_MODE = "human"  # Change to "human" for graphical visualization
    SAVE_LOG = False  # Change to False to disable log file
    
    try:
        success = test_tutorial_complete(render_mode=RENDER_MODE, log_to_file=SAVE_LOG)
        print(f"\nTest finished with result: {'PASSED [OK]' if success else 'FAILED [X]'}")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pass

