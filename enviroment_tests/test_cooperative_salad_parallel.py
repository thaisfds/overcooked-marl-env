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
            "agent_0": 2,  # LEFT: (2,1) -> (1,1) [heading to tomato]
            "agent_1": 1   # DOWN: (2,3) -> (2,4) [arrived at lettuce!]
        }),
        ("PARALLEL: Agent 0 continues, Agent 1 turns", {
            "agent_0": 1,  # DOWN: (1,1) -> (1,2)
            "agent_1": 1   # DOWN: turn South to point to (2,4)
        }),
        ("PARALLEL: Agent 0 continues, Agent 1 picks lettuce", {
            "agent_0": 1,  # DOWN: (1,2) -> (1,3)
            "agent_1": 4   # INTERACT: pick lettuce! [First ingredient collected!]
        }),
        ("PARALLEL: Agent 0 turns, Agent 1 returns", {
            "agent_0": 1,  # DOWN: turn South to point to (1,4)
            "agent_1": 0   # UP: (2,4) -> (2,3)
        }),
        ("PARALLEL: Agent 0 picks tomato, Agent 1 continues back", {
            "agent_0": 4,  # INTERACT: pick tomato! [Second ingredient collected!]
            "agent_1": 0   # UP: (2,3) -> (2,2)
        }),
        
        # ===== PHASE 2a: Agent 0 Chops Tomato =====
        ("PARALLEL: Agent 0 goes to chopping, Agent 1 clears path", {
            "agent_0": 0,  # UP: (1,3) -> (1,2)
            "agent_1": 3   # RIGHT: (2,2) -> (3,2) [CLEARING PATH for agent_0!]
        }),
        ("PARALLEL: Agent 0 reaches chopping position, Agent 1 waits", {
            "agent_0": 0,  # UP: (1,2) -> (1,1) [adjacent to chopping!]
            "agent_1": 2   # LEFT: (3,2) -> (2,2) [moving but NOT interacting with items]
        }),
        ("PARALLEL: Agent 0 faces chopping board, Agent 1 waits", {
            "agent_0": 0,  # UP: turn North to point to (1,0) chopping board
            "agent_1": 3   # RIGHT: (2,2) -> (3,2) [moving back and forth, no interact!]
        }),
        ("PARALLEL: Agent 0 cuts tomato 1/3, Agent 1 waits", {
            "agent_0": 4,  # INTERACT: cut tomato 1/3
            "agent_1": 2   # LEFT: (3,2) -> (2,2) [keep moving to avoid dropping items]
        }),
        ("PARALLEL: Agent 0 cuts tomato 2/3, Agent 1 waits", {
            "agent_0": 4,  # INTERACT: cut tomato 2/3
            "agent_1": 3   # RIGHT: (2,2) -> (3,2) [keep moving]
        }),
        ("PARALLEL: Agent 0 cuts tomato 3/3 READY!, Agent 1 prepares", {
            "agent_0": 4,  # INTERACT: cut tomato 3/3 - READY!
            "agent_1": 2   # LEFT: (3,2) -> (2,2) [moving toward chopping]
        }),
        
        # ===== PHASE 2b: Transition - Agent 0 leaves, Agent 1 enters =====
        ("PARALLEL: Agent 0 exits chopping DOWN, Agent 1 approaches", {
            "agent_0": 1,  # DOWN: (1,1) -> (1,2) [leaving chopping area going down]
            "agent_1": 2   # LEFT: (2,2) -> (1,2) CONFLICT! Will stay at (2,2)
        }),
        ("PARALLEL: Agent 0 moves right, Agent 1 to chopping", {
            "agent_0": 3,  # RIGHT: (1,2) -> (2,2) CONFLICT with Agent_1!
            "agent_1": 2   # LEFT: (2,2) -> (1,2)
        }),
        ("PARALLEL: Agent 0 continues, Agent 1 reaches chopping", {
            "agent_0": 3,  # RIGHT: (2,2) -> (3,2) [now clear]
            "agent_1": 0   # UP: (1,2) -> (1,1) [adjacent to chopping!]
        }),
        
        # ===== PHASE 2c: Agent 1 Chops Lettuce =====
        ("PARALLEL: Agent 0 waits, Agent 1 faces chopping", {
            "agent_0": 3,  # RIGHT: collide with wall, stay at (3,2)
            "agent_1": 0   # UP: turn North to point to (1,0) chopping board
        }),
        ("PARALLEL: Agent 0 waits, Agent 1 cuts lettuce 1/3", {
            "agent_0": 0,  # UP: (3,2) -> (3,1) [going to plate dispenser area]
            "agent_1": 4   # INTERACT: cut lettuce 1/3
        }),
        ("PARALLEL: Agent 0 waits, Agent 1 cuts lettuce 2/3", {
            "agent_0": 0,  # UP: collide, turn North to point to (3,0)
            "agent_1": 4   # INTERACT: cut lettuce 2/3
        }),
        ("PARALLEL: Agent 0 prepares, Agent 1 cuts lettuce 3/3 READY!", {
            "agent_0": 2,  # LEFT: (3,1) -> (2,1) [clearing path before plate pickup]
            "agent_1": 4   # INTERACT: cut lettuce 3/3 - READY!
        }),
        
        # ===== PHASE 3a: Agent 0 Gets Plate and Combines with Tomato =====
        ("PARALLEL: Agent 0 to plate position, Agent 1 exits chopping", {
            "agent_0": 3,  # RIGHT: (2,1) -> (3,1) [adjacent to plate dispenser!]
            "agent_1": 1   # DOWN: (1,1) -> (1,2) [leaving chopping]
        }),
        ("PARALLEL: Agent 0 faces plate, Agent 1 moves", {
            "agent_0": 0,  # UP: turn North to point to (3,0) plate dispenser
            "agent_1": 3   # RIGHT: (1,2) -> (2,2) [moving away]
        }),
        ("PARALLEL: Agent 0 gets plate+tomato, Agent 1 continues", {
            "agent_0": 4,  # INTERACT: pick plate and AUTO-combine with READY tomato!
            "agent_1": 3   # RIGHT: (2,2) -> (3,2) [positioning]
        }),
        
        # ===== PHASE 3b: Agent 0 Places Plate+Tomato on Counter =====
        ("PARALLEL: Agent 0 moves toward counter, Agent 1 clears", {
            "agent_0": 1,  # DOWN: (3,1) -> (3,2)
            "agent_1": 1   # DOWN: (3,2) -> (3,3) [moving out of the way]
        }),
        ("PARALLEL: Agent 0 to counter, Agent 1 waits", {
            "agent_0": 1,  # DOWN: (3,2) -> (3,3) CONFLICT! Will stay or resolve
            "agent_1": 2   # LEFT: (3,3) -> (2,3) [clearing space]
        }),
        ("PARALLEL: Agent 0 faces counter, Agent 1 waits", {
            "agent_0": 1,  # DOWN: turn South to point to (3,4) counter
            "agent_1": 0   # UP: (2,3) -> (2,2) [staying clear]
        }),
        ("PARALLEL: Agent 0 places plate+tomato, Agent 1 moves closer", {
            "agent_0": 4,  # INTERACT: place plate+tomato on counter at (3,4)
            "agent_1": 3   # RIGHT: (2,2) -> (3,2) [getting closer to counter]
        }),
        
        # ===== PHASE 4: Agent 1 Picks Plate+Tomato, Combines with Lettuce, and Delivers =====
        ("PARALLEL: Agent 0 backs away, Agent 1 waits", {
            "agent_0": 2,  # LEFT: (3,3) -> (2,3) [moving left out of column 3]
            "agent_1": 2   # LEFT: (3,2) -> (2,2) [moving left to clear path]
        }),
        ("PARALLEL: Agent 0 continues away, Agent 1 waits", {
            "agent_0": 2,  # LEFT: (2,3) -> (1,3) [continuing away]
            "agent_1": 1   # DOWN: (2,2) -> (2,3) [moving down]
        }),
        ("PARALLEL: Agent 0 far away, Agent 1 to counter column", {
            "agent_0": 0,  # UP: (1,3) -> (1,2) [moving away]
            "agent_1": 3   # RIGHT: (2,3) -> (3,3) [adjacent to counter!]
        }),
        ("PARALLEL: Agent 0 away, Agent 1 faces counter", {
            "agent_0": 0,  # UP: (1,2) -> (1,1) [very far away]
            "agent_1": 1   # DOWN: turn South to point to (3,4) counter
        }),
        ("PARALLEL: Agent 0 away, Agent 1 picks plate and combines!", {
            "agent_0": 2,  # LEFT: stay at (1,1)
            "agent_1": 4   # INTERACT: pick plate+tomato, AUTO-combine with READY lettuce!
        }),
        ("PARALLEL: Agent 0 stays, Agent 1 moves to delivery", {
            "agent_0": 2,  # LEFT: stay still
            "agent_1": 0   # UP: (3,3) -> (3,2) [heading to delivery]
        }),
        ("PARALLEL: Agent 0 stays, Agent 1 faces delivery", {
            "agent_0": 2,  # LEFT: stay still
            "agent_1": 3   # RIGHT: turn East to point to (4,2) delivery
        }),
        ("PARALLEL: Agent 1 DELIVERS COMPLETE SALAD!!!", {
            "agent_0": 2,  # LEFT: (celebrating!)
            "agent_1": 4   # INTERACT: DELIVER COMPLETE SALAD!!!
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
    RENDER_MODE = "human"  # Change to "rgb_array" for graphical visualization
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

