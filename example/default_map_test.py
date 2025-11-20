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


def test_cooperative_salad_parallel(render_mode="human", log_to_file=True):
    """
    Test where two agents cooperate with MAXIMUM PARALLELISM.
    
    Args:
        render_mode: "ascii" for ASCII console (default) or "human" for Pygame graphics
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
    print("PARALLEL COOPERATION TEST - TEAM SALAD")
    print("="*70)
    print("\nThis test demonstrates MAXIMUM PARALLELISM cooperation:")
    print("  • PHASE 1: Both collect ingredients simultaneously")
    print("  • PHASE 2: Coordinate chopping (minimize idle time)")
    print("  • PHASE 3: Parallel assembly and counter placement")
    print("  • PHASE 4: Final pickup and delivery")
    print("="*70 + "\n")
    
    env = OvercookedMARL(render_mode=render_mode, verbose=True)
    obs, info = env.reset()
    
    # Render initial state
    env.render()
        time.sleep(0.5 if render_mode == "human" else 0.5)
    
    # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Interact
    # Agent 0 starts at (2,1)
    # Agent 1 starts at (2,3)
    
    actions = [
        # ===================================================================
        # PHASE 1: PARALLEL Ingredient Collection
        # Both agents move to dispensers and collect ingredients simultaneously
        # ===================================================================
        
        ("PHASE 1 START: Parallel Ingredient Collection", {}),
        
        ("Both agents move toward dispensers", {
            "agent_0": 2,  # Agent 0 moves left
            "agent_1": 1   # Agent 1 moves down
        }),
        ("Agent 0 continues, Agent 1 gets lettuce", {
            "agent_0": 1,  # Agent 0 moves down
            "agent_1": 4   # Agent 1 interacts with lettuce dispenser
        }),
        ("Agent 0 continues, Agent 1 moves up", {
            "agent_0": 1,  # Agent 0 moves down
            "agent_1": 0   # Agent 1 moves up
        }),
        ("Agent 0 gets tomato, Agent 1 moves left", {
            "agent_0": 4,  # Agent 0 interacts with tomato dispenser
            "agent_1": 2   # Agent 1 moves left
        }),
        
        # ===================================================================
        # PHASE 2: Coordinated Processing
        # Minimize wait time by processing while the other agent moves
        # ===================================================================
        
        ("PHASE 2 START: Coordinated Processing", {}),
        
        ("Agent 1 moves to chopping board area", {
            "agent_1": 0   # Agent 1 moves up
        }),
        ("Agent 0 moves up, Agent 1 places lettuce", {
            "agent_0": 0,  # Agent 0 moves up
            "agent_1": 4   # Agent 1 places lettuce on chopping board
        }),
        ("Agent 1 chops lettuce (1/3)", {
            "agent_1": 4
        }),
        ("Agent 1 chops lettuce (2/3)", {
            "agent_1": 4
        }),
        ("Agent 1 chops lettuce (3/3)", {
            "agent_1": 4
        }),
        ("Agent 1 picks ready lettuce", {
            "agent_1": 4
        }),
        ("Agent 1 moves right", {
            "agent_1": 3
        }),
        ("Agent 0 moves up, Agent 1 continues right", {
            "agent_0": 0,
            "agent_1": 3
        }),
        ("Agent 0 places tomato, Agent 1 moves up", {
            "agent_0": 4,
            "agent_1": 0
        }),
        ("Agent 0 chops tomato (1/3), Agent 1 places lettuce on counter", {
            "agent_0": 4,
            "agent_1": 4
        }),
        ("Agent 0 chops tomato (2/3), Agent 1 moves right", {
            "agent_0": 4,
            "agent_1": 3
        }),
        ("Agent 0 chops tomato (3/3), Agent 1 picks lettuce from counter", {
            "agent_0": 4,
            "agent_1": 4
        }),
        
        # ===================================================================
        # PHASE 3: Parallel Assembly
        # Agent 0 gets plate while Agent 1 positions for transfer
        # ===================================================================
        
        ("PHASE 3 START: Parallel Assembly", {}),
        
        ("Agent 0 picks tomato, Agent 1 moves down", {
            "agent_0": 4,
            "agent_1": 1
        }),
        ("Both agents move", {
            "agent_0": 3,  # Agent 0 moves right
            "agent_1": 1   # Agent 1 moves down
        }),
        ("Both agents continue moving", {
            "agent_0": 3,  # Agent 0 moves right
            "agent_1": 1   # Agent 1 moves down
        }),
        ("Agent 0 moves right", {
            "agent_0": 3
        }),
        ("Agent 0 gets plate", {
            "agent_0": 4
        }),
        
        # ===================================================================
        # PHASE 4: Final Delivery
        # Combine ingredients and deliver the salad
        # ===================================================================
        
        ("PHASE 4 START: Final Delivery", {}),
        
        ("Agent 0 gets plate", {
            "agent_0": 4
        }),
        ("Agent 0 moves down", {
            "agent_0": 1
        }),
        ("Agent 0 moves right", {
            "agent_0": 3
        }),
        ("Agent 0 combines plate + tomato", {
            "agent_0": 4
        }),
    ]
    
    step_count = 0
    recipes_delivered = 0
    
    for desc, action_dict in actions:
        # Skip empty description entries (used for spacing)
        if not desc:
            continue
            
        # Check for special announcement messages (no actions)
        if not action_dict:
            print(f"\n{'='*70}")
            print(f">>> {desc}")
            print('='*70)
            time.sleep(0.2 if render_mode == "human" else 0.3)
            continue
        
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
            print(f"{agent_id}: Inventory = {inv:40s} | Step Reward = {rew:+.3f} | Total = {acc_rew:+.3f}")
        
        # Show recipe progress
        delivered = len(info['agent_0']['delivered_recipes'])
        remaining = info['agent_0']['dishes_remaining']
        if delivered > recipes_delivered:
            recipes_delivered = delivered
            print(f"\n*** RECIPE DELIVERED! Total: {delivered} ***")
        
        # Delay for visualization
        time.sleep(0.2 if render_mode == "human" else 0.15)
        
        # Check if completed
        if delivered > 0:
            print("\n" + "="*70)
            print("*** SALAD DELIVERED SUCCESSFULLY! ***")
            print("="*70)
            break
    
    # Final summary
    print("\n" + "="*70)
    print("PARALLEL COOPERATION TEST - FINAL SUMMARY")
    print("="*70)
    print(f"Total steps taken: {step_count}")
    print(f"Recipes delivered: {info['agent_0']['delivered_recipes']}")
    print(f"Remaining orders: {info['agent_0']['dishes_remaining']}")
    print(f"Agent 0 total reward: {env.state['accumulated_rewards']['agent_0']:+.2f}")
    print(f"Agent 1 total reward: {env.state['accumulated_rewards']['agent_1']:+.2f}")
    print("="*70)
    print("\nCOMPARISON TO SEQUENTIAL VERSION:")
    print(f"  Sequential test: ~38 steps")
    print(f"  Parallel test:   {step_count} steps")
    if step_count < 38:
        print(f"  Steps saved:     {38 - step_count} steps ({((38-step_count)/38*100):.1f}% faster!)")
    print("="*70)
    
    # Check success
    success = len(info['agent_0']['delivered_recipes']) >= 1
    
    if success:
        print("\n" + "="*70)
        print("✓✓✓ TEST PASSED! ✓✓✓")
        print("Parallel cooperation tested successfully:")
        print("  ✓ Simultaneous ingredient collection")
        print("  ✓ Coordinated processing to minimize wait time")
        print("  ✓ Parallel assembly workflow")
        print("  ✓ Successful recipe delivery")
        print("="*70)
    else:
        print("\n[FAILED] Recipe was not delivered.")
        print(f"Expected: 1+ recipe | Delivered: {len(info['agent_0']['delivered_recipes'])}")
    
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
    print("\n" + "="*70)
    print(" PARALLEL COOPERATION TEST - TEAM SALAD")
    print("="*70)
    print("\nThis test demonstrates TRUE PARALLEL cooperation where")
    print("both agents work simultaneously on different tasks:")
    print("  • Simultaneous ingredient collection")
    print("  • Coordinated processing")
    print("  • Parallel assembly workflow")
    print("  • Complete recipe delivery")
    print("\nConfiguration:")
    print('  - RENDER_MODE: "ascii" (ASCII) or "human" (Pygame)')
    print('  - SAVE_LOG: True (save to logs/) or False (console only)')
    print("="*70)
    
    # Configuration
    RENDER_MODE = "human"  # Change to "human" for graphical visualization
    SAVE_LOG = True  # Change to False to disable log file
    
    try:
        success = test_cooperative_salad_parallel(render_mode=RENDER_MODE, log_to_file=SAVE_LOG)
        print(f"\nTest finished with result: {'PASSED ✓' if success else 'FAILED ✗'}")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pass

