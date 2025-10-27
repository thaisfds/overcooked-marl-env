"""
Parallel Cooperation Test: Two agents make a salad with MAXIMUM PARALLELISM.

PARALLEL Strategy - Both agents work simultaneously:
1. PHASE 1: Both agents collect ingredients at the SAME TIME
   - Agent 0: Gets tomato from (1,4)
   - Agent 1: Gets lettuce from (2,4)
2. PHASE 2: Coordinated processing (minimize wait time)
   - Agent 0: Cuts tomato while Agent 1 positions nearby
   - Agent 1: Cuts lettuce while Agent 0 moves to plate dispenser
3. PHASE 3: Parallel assembly
   - Agent 0: Gets plate and adds tomato while Agent 1 finishes cutting
   - Agent 0: Places on counter while Agent 1 moves to counter
4. PHASE 4: Final delivery
   - Agent 1: Picks up plate+tomato, adds lettuce, delivers
   - Agent 0: Moves away to avoid blocking

Expected: ~33 steps (vs 38 steps in sequential version - 13% faster!)

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
import os
import sys
from datetime import datetime
from overcooked_marl_env.env import OvercookedMARL


class Logger:
    """Duplica output para console e arquivo."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def test_cooperative_salad_parallel(render_mode="human", log_to_file=True):
    """
    Test where two agents cooperate with MAXIMUM PARALLELISM.
    
    Args:
        render_mode: "human" for ASCII console (default) or "rgb_array" for Pygame graphics
        log_to_file: If True, saves output to logs/test_parallel_TIMESTAMP.txt
    """
    # Setup logging to file if requested
    logger = None
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"test_parallel_{timestamp}.txt")
        
        logger = Logger(log_file)
        sys.stdout = logger
        print(f"[Logging to: {log_file}]\n")
    
    print("\n" + "="*70)
    print("TEST: PARALLEL Cooperation to Make Salad")
    print("="*70)
    print("\nParallel Strategy:")
    print("  Phase 1: BOTH collect ingredients simultaneously")
    print("  Phase 2: Coordinate chopping (minimize idle time)")
    print("  Phase 3: Parallel assembly and counter placement")
    print("  Phase 4: Final pickup and delivery")
    print("="*70 + "\n")
    
    env = OvercookedMARL(render_mode=render_mode, verbose=False)
    obs, info = env.reset()
    
    # Render initial state
    env.render()
    time.sleep(1.0 if render_mode == "rgb_array" else 0.5)
    
    # Agent 0 starts at (2,1)
    # Agent 1 starts at (2,3)
    
    actions = [
        # ===== PHASE 1: PARALLEL Ingredient Collection =====
        ("PARALLEL: Both move toward dispensers", {
            "agent_0": 2, 
            "agent_1": 1   
        }),
        ("PARALLEL: 0 moves and 1 interact", {
            "agent_0": 1, 
            "agent_1": 4   
        }),
        ("PARALLEL: 0 moves and 1 moves", {
            "agent_0": 1, 
            "agent_1": 0   
        }),
        ("PARALLEL: 0 interact and 1 moves", {
            "agent_0": 4, 
            "agent_1": 2   
        }),
        ("PARALLEL: 0 wait and 1 moves", {
            "agent_1": 0   
        }),
        ("PARALLEL: 0 moves and 1 interact", {
            "agent_0": 0,
            "agent_1": 4   
        }),
        ("PARALLEL: 0 wait and 1 interact", {
            "agent_1": 4   
        }),
        ("PARALLEL: 0 wait and 1 interact", {
            "agent_1": 4   
        }),
        ("PARALLEL: 0 wait and 1 interact", {
            "agent_1": 4   
        }),
        ("PARALLEL: 0 wait and 1 interact", {
            "agent_1": 4   
        }),
        ("PARALLEL: 0 wait and 1 moves", {
            "agent_1": 3   
        }),
        ("PARALLEL: 0 moves and 1 moves", {
            "agent_0": 0,
            "agent_1": 3   
        }),
        ("PARALLEL: 0 interact and 1 moves", {
            "agent_0": 4,
            "agent_1": 0   
        }),
        ("PARALLEL: 0 interact and 1 interact", {
            "agent_0": 4,
            "agent_1": 4   
        }),
        ("PARALLEL: 0 interact and 1 moves", {
            "agent_0": 4,
            "agent_1": 3   
        }),
        ("PARALLEL: 0 interact and 1 interact", {
            "agent_0": 4,
            "agent_1": 4   
        }),
        ("PARALLEL: 0 interact and 1 moves", {
            "agent_0": 4,
            "agent_1": 1   
        }),
        ("PARALLEL: 0 moves and 1 moves", {
            "agent_0": 3,
            "agent_1": 1   
        }),
        ("PARALLEL: 0 moves and 1 moves", {
            "agent_0": 3,
            "agent_1": 1   
        }),
        ("PARALLEL: 0 moves and 1 wait", {
            "agent_0": 3,
        }),
        ("PARALLEL: 0 interact and 1 wait", {
            "agent_0": 4,
        }),
        ("PARALLEL: 0 moves and 1 wait", {
            "agent_0": 1,
        }),
        ("PARALLEL: 0 moves and 1 wait", {
            "agent_0": 3,
        }),
        ("PARALLEL: 0 interact and 1 wait", {
            "agent_0": 4,
        }),

    ]
    
    step_count = 0
    for desc, action_dict in actions:
        step_count += 1
        print(f"\n{'='*70}")
        print(f"STEP {step_count}: {desc}")
        print('='*70)
        
        obs, rewards, dones, truncs, info = env.step(action_dict)
        env.render()
        
        # Show agent information
        for agent_id in ["agent_0", "agent_1"]:
            inv = info[agent_id]['inventory']
            rew = rewards[agent_id]
            acc_rew = env.state["accumulated_rewards"][agent_id]
            print(f"{agent_id}: Inventory = {inv:30s} | Step Reward = {rew:+.3f} | Total = {acc_rew:+.3f}")
        
        # Delay for visualization (pygame events handled by env.render())
        time.sleep(0.3 if render_mode == "rgb_array" else 0.15)
        
        # Check if completed
        if info['agent_0']['dishes_remaining'] < 3:
            print("\n" + "="*70)
            print("*** SALAD DELIVERED SUCCESSFULLY! ***")
            print("="*70)
            break
    
    print("\n" + "="*70)
    print(f"PARALLEL Test completed in {step_count} steps!")
    print(f"Delivered recipes: {info['agent_0']['delivered_recipes']}")
    print(f"Remaining dishes: {info['agent_0']['dishes_remaining']}")
    print("\n" + "="*70)
    print("COMPARISON TO SEQUENTIAL VERSION:")
    print(f"  Sequential test: 38 steps")
    print(f"  Parallel test:   {step_count} steps")
    print(f"  Steps saved:     {38 - step_count} steps ({((38-step_count)/38*100):.1f}% faster!)")
    print("="*70)
    
    # Check success
    success = False
    if len(info['agent_0']['delivered_recipes']) > 0:
        print("\n[OK] TEST PASSED! Salad delivered successfully with parallelism!")
        success = True
    else:
        print("\n[FAILED] No recipe was delivered.")
        success = False
    
    # Keep window open for rgb_array mode
    if render_mode == "rgb_array":
        print("\nFinal state rendered. Close the Pygame window to exit.")
        print("(Window will auto-close in 10 seconds)")
        
        # Just wait - env.render() handles pygame events internally
        for _ in range(100):  # 10 seconds total
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
    print("\n" + "="*70)
    print(" PARALLEL COOPERATION TEST - TEAM SALAD")
    print("="*70)
    print("\nThis test demonstrates TRUE PARALLEL cooperation where")
    print("both agents work simultaneously on different tasks.")
    print("\nConfiguration options:")
    print('  - RENDER_MODE: "human" (ASCII console) or "rgb_array" (Pygame graphics)')
    print('  - SAVE_LOG: True (saves to logs/) or False (console only)')
    print("="*70)
    
    # Easy configuration - just change these lines!
    RENDER_MODE = "rgb_array"  # Change to "rgb_array" for graphical visualization
    SAVE_LOG = False  # Change to False to disable log file creation
    
    try:
        success = test_cooperative_salad_parallel(render_mode=RENDER_MODE, log_to_file=SAVE_LOG)
        print(f"\nParallel test finished with result: {'PASSED' if success else 'FAILED'}")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Environment cleanup handled by env.close() in the test function
        pass

