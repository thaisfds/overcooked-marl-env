import pygame
import time
import sys
import os
import argparse
from overcooked_marl_env.env import OvercookedMARL

def play(map_path=None, custom_recipes=None):
    # Initialize Pygame display and sound modules
    pygame.init()
    
    # Print instruction manual to console
    print("\n" + "="*60)
    print("      OVERCOOKED MARL KEYBOARD PLAYTEST RUNNER")
    print("="*60)
    print("Controls:")
    print("  • Agent 0 (Blue Agent):")
    print("    - Move: W (Up), A (Left), S (Down), D (Right)")
    print("    - Interact: Spacebar")
    print("  • Agent 1 (Orange Agent):")
    print("    - Move: Up, Down, Left, Right Arrow keys")
    print("    - Interact: Enter / Return key")
    print("  • General:")
    print("    - Reset Environment: R")
    print("    - Exit: Escape key or close window")
    print("="*60)
    
    # Instantiate the environment in human mode (Pygame)
    env = OvercookedMARL(
        render_mode="human",
        map_path=map_path,
        verbose=True,
        custom_recipes_path=custom_recipes
    )
    
    obs, info = env.reset()
    env.render()
    
    clock = pygame.time.Clock()
    running = True
    
    # Interaction triggers from KEYDOWN events (prevent rapid repeat)
    interact_0 = False
    interact_1 = False
    
    while running:
        # Check event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    print("\n[GAME] Resetting environment manually...")
                    obs, info = env.reset()
                    env.render()
                elif event.key == pygame.K_SPACE:
                    interact_0 = True
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    interact_1 = True
        
        # Check continuous key state for movement (allows holding key to walk)
        keys = pygame.key.get_pressed()
        
        actions = {}
        
        # Agent 0 movement/interaction decision
        if keys[pygame.K_w]:
            actions["agent_0"] = 0  # UP
        elif keys[pygame.K_s]:
            actions["agent_0"] = 1  # DOWN
        elif keys[pygame.K_a]:
            actions["agent_0"] = 2  # LEFT
        elif keys[pygame.K_d]:
            actions["agent_0"] = 3  # RIGHT
        elif interact_0:
            actions["agent_0"] = 4  # INTERACT
            interact_0 = False
            
        # Agent 1 movement/interaction decision
        if keys[pygame.K_UP]:
            actions["agent_1"] = 0  # UP
        elif keys[pygame.K_DOWN]:
            actions["agent_1"] = 1  # DOWN
        elif keys[pygame.K_LEFT]:
            actions["agent_1"] = 2  # LEFT
        elif keys[pygame.K_RIGHT]:
            actions["agent_1"] = 3  # RIGHT
        elif interact_1:
            actions["agent_1"] = 4  # INTERACT
            interact_1 = False
            
        # Consume any unhandled interaction triggers
        interact_0 = False
        interact_1 = False
        
        # Step the environment only if an agent takes an action
        if actions:
            obs, rewards, terminations, truncations, infos = env.step(actions)
            env.render()
            
            # Print state changes to console
            for agent_id in env.agents:
                if agent_id in actions:
                    inv = infos[agent_id]['inventory']
                    act_name = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT"][actions[agent_id]]
                    print(f"{agent_id}: Action={act_name} | Inventory={inv} | Reward={rewards[agent_id]:+.1f}")
            
            # Reset automatically on completion
            if any(terminations.values()) or any(truncations.values()):
                print("\n[GAME] Episode finished! Auto-resetting in 1.5 seconds...")
                time.sleep(1.5)
                obs, info = env.reset()
                env.render()
        else:
            # Maintain screen responsiveness and window events even when idle
            env.render()
            
        # Frame limit of 8 steps/second feels natural for grid movement
        clock.tick(8)
        
    env.close()
    pygame.quit()
    print("[GAME] Play runner terminated. Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Playtest for Overcooked MARL")
    parser.add_argument("--map", default=None, help="Path to map text file")
    parser.add_argument("--recipes", default=None, help="Path to recipes JSON file")
    args = parser.parse_args()
    
    # Try using custom map template by default if it exists
    map_file = args.map
    if map_file is None:
        template_map = "example/custom_map_template.txt"
        if os.path.exists(template_map):
            map_file = template_map
            print(f"Auto-selected custom map template: {template_map}")
            
    play(map_file, args.recipes)
