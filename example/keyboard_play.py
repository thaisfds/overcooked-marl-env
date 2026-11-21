import pygame
import time
import sys
import os
import argparse
from pathlib import Path
from overcooked_marl_env.env import OvercookedMARL

def resolve_map_path(map_name_or_path: str = None) -> str:
    """
    Resolve o caminho do mapa de forma inteligente:
    1. Caminho direto relativo ou absoluto
    2. Nome do arquivo dentro de overcooked_marl_env/map/maps/
    3. Adiciona automaticamente a extensão .txt se omitida
    """
    project_root = Path(__file__).resolve().parent.parent
    maps_dir = project_root / "overcooked_marl_env" / "map" / "maps"

    if not map_name_or_path:
        default_map = maps_dir / "default_map.txt"
        return str(default_map) if default_map.exists() else "default_map.txt"

    candidates = [
        Path(map_name_or_path),
        Path(f"{map_name_or_path}.txt"),
        maps_dir / map_name_or_path,
        maps_dir / f"{map_name_or_path}.txt",
        project_root / map_name_or_path,
        project_root / f"{map_name_or_path}.txt",
    ]

    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())

    # Se não encontrar, lista mapas disponíveis para orientar o usuário
    available = [f.name for f in maps_dir.glob("*.txt")] if maps_dir.exists() else []
    print(f"\n[AVISO] Mapa '{map_name_or_path}' não foi encontrado!")
    if available:
        print(f"[INFO] Mapas disponíveis em '{maps_dir.name}/': {', '.join(available)}")
    return map_name_or_path

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

def play(map_path=None, custom_recipes=None, num_agents=1):
    # Initialize Pygame display and sound modules
    pygame.init()
    
    # Instantiate the environment in human mode (Pygame)
    env = OvercookedMARL(
        render_mode="human",
        map_path=map_path,
        num_fixed_agents=num_agents,
        verbose=True,
        custom_recipes_path=custom_recipes
    )
    env.renderer.handle_window_events = False
    
    # Print instruction manual to console based on active agents
    print("\n" + "="*60)
    print("      OVERCOOKED MARL KEYBOARD PLAYTEST RUNNER")
    print("="*60)
    print("Controls:")
    if "agent_0" in env.agents:
        print("  • Agent 0 (Blue Agent):")
        print("    - Move: W (Up), A (Left), S (Down), D (Right)")
        print("    - Interact: Spacebar")
    if "agent_1" in env.agents:
        print("  • Agent 1 (Orange Agent):")
        print("    - Move: Up, Down, Left, Right Arrow keys")
        print("    - Interact: Enter / Return key")
    print("  • General:")
    print("    - Reset Environment: R")
    print("    - Exit: Escape key or close window")
    print("="*60)
    
    obs, info = env.reset()
    env.render()
    run_countdown(env)
    
    clock = pygame.time.Clock()
    running = True
    
    # Interaction triggers from KEYDOWN events (prevent rapid repeat)
    interact_0 = False
    interact_1 = False
    

    
    while running:
        current_time = pygame.time.get_ticks()
        
        # Check event queue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                env.renderer.handle_resize(event.w, event.h, env.map_size)
                env.render()
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
        if "agent_0" in env.agents:
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
        if "agent_1" in env.agents:
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
    parser.add_argument("--map", "-m", default=None, help="Name or path of the map (e.g. tutorial, default_map.txt)")
    parser.add_argument("--recipes", "-r", default=None, help="Path to recipes JSON file")
    parser.add_argument("--agents", "-a", type=int, default=1, help="Number of agents to play with (default: 1, up to 4)")
    parser.add_argument("--list-maps", "-l", action="store_true", help="List all available maps in the environment")
    args = parser.parse_args()

    # List maps if requested
    if args.list_maps:
        maps_dir = Path(__file__).resolve().parent.parent / "overcooked_marl_env" / "map" / "maps"
        available = sorted([f.name for f in maps_dir.glob("*.txt")]) if maps_dir.exists() else []
        print("\n🗺️  MAPAS DISPONÍVEIS:")
        for m in available:
            print(f"  • {m}  ->  (use: --map {m.replace('.txt', '')})")
        print("\nExemplo de uso:")
        print("  python example/keyboard_play.py --map tutorial --agents 2\n")
        sys.exit(0)

    # Resolves map file automatically
    map_file = resolve_map_path(args.map)
    print(f"\n[CONFIG] Mapa selecionado: {map_file}")
    print(f"[CONFIG] Agentes: {args.agents}\n")

    play(map_file, args.recipes, num_agents=args.agents)
