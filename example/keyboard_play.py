import pygame
import time
import sys
import os
import argparse
from overcooked_marl_env.env import OvercookedMARL

def run_countdown(env):
    """Run a 3, 2, 1 countdown overlay on the game screen without flickering."""
    # Render environment once to ensure window is populated
    env.render()
    screen = env.renderer.display_window
    if screen is None:
        return
        
    # Capture the rendered game screen as a static background surface
    background = screen.copy()
    
    # Track debounced resizing variables
    resize_pending = False
    target_w, target_h = None, None
    last_resize_time = 0
    
    for count in ["3", "2", "1", "START!"]:
        start_ticks = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_ticks < 1000:
            current_time = pygame.time.get_ticks()
            
            # Process events so window remains responsive and Esc/Close/Resize work
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    target_w = event.w
                    target_h = event.h
                    resize_pending = True
                    last_resize_time = current_time
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                        infoObject = pygame.display.Info()
                        env.renderer.handle_resize(infoObject.current_w, infoObject.current_h, env.map_size)
                        env.render()
                        screen = env.renderer.display_window
                        if screen is not None:
                            background = screen.copy()
            
            # Apply pending resize if debounced (no events for 150ms)
            if resize_pending and current_time - last_resize_time > 150:
                env.renderer.handle_resize(target_w, target_h, env.map_size)
                resize_pending = False
                env.render()
                screen = env.renderer.display_window
                if screen is not None:
                    background = screen.copy()
            
            # Blit the static background (no double rendering/flipping)
            screen.blit(background, (0, 0))
            
            # Create overlay surface for a transparent black circle in the center
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            
            # Circle dimensions
            center_x, center_y = screen.get_rect().center
            circle_radius = 80
            
            # Draw semi-transparent black circle in the center (black with 180 alpha)
            pygame.draw.circle(overlay, (0, 0, 0, 180), (center_x, center_y), circle_radius)
            screen.blit(overlay, (0, 0))
            
            # Draw countdown text centered inside the circle
            font = pygame.font.Font(None, 100)
            text_color = (255, 215, 0) if count != "START!" else (50, 205, 50)  # Golden or green
            text_surf = font.render(count, True, text_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            screen.blit(text_surf, text_rect)
            
            # Flip ONLY ONCE per frame update
            pygame.display.flip()
            
            pygame.time.wait(20)

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
    run_countdown(env)
    
    clock = pygame.time.Clock()
    running = True
    
    # Interaction triggers from KEYDOWN events (prevent rapid repeat)
    interact_0 = False
    interact_1 = False
    
    # Track debounced resizing variables
    resize_pending = False
    target_w, target_h = None, None
    last_resize_time = 0
    
    while running:
        current_time = pygame.time.get_ticks()
        
        # Check event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                target_w = event.w
                target_h = event.h
                resize_pending = True
                last_resize_time = current_time
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                    infoObject = pygame.display.Info()
                    env.renderer.handle_resize(infoObject.current_w, infoObject.current_h, env.map_size)
                    env.render()
                elif event.key == pygame.K_r:
                    print("\n[GAME] Resetting environment manually...")
                    obs, info = env.reset()
                    env.render()
                    run_countdown(env)
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
        
        # Apply pending resize if debounced (no events for 150ms)
        if resize_pending and current_time - last_resize_time > 150:
            env.renderer.handle_resize(target_w, target_h, env.map_size)
            resize_pending = False
            env.render()
            
        # Step the environment unconditionally so time passes even when idle
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
            run_countdown(env)
            
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
        template_map = "overcooked_marl_env/map/maps/tutorial.txt"
        if os.path.exists(template_map):
            map_file = template_map
            print(f"Auto-selected tutorial map: {template_map}")
            
    play(map_file, args.recipes)
