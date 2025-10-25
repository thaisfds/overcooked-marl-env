"""
Visualization test for the Overcooked MARL environment.
Tests both ASCII and RGB rendering modes.
"""

import os
import sys
import time
import numpy as np

# Check if matplotlib is available for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not found. RGB array visualization will be skipped.")

# Check if pygame is available for interactive visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Pygame not found. Interactive visualization will be skipped.")

# Import the environment class
from overcooked_marl_env.env import OvercookedMARL

def test_ascii_rendering():
    """Test the ASCII rendering mode (human mode)."""
    print("\n--- Testing ASCII Rendering ---")
    
    # Create environment with human rendering
    env = OvercookedMARL(render_mode="human", verbose=False)
    
    # Reset environment
    observations, info = env.reset(seed=42)
    
    # Action mapping for display
    action_names = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right",
        4: "Interact"
    }
    
    # Render initial state
    print("\nInitial State:")
    env.render()
    
    # Take a few random steps and render each
    print("\nAfter random actions:")
    for i in range(5):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Format actions for display
        formatted_actions = {}
        for agent, action in actions.items():
            # Convert numpy int to Python int if needed
            if hasattr(action, 'item'):
                action_int = action.item()
            else:
                action_int = int(action)
            
            formatted_actions[agent] = action_names.get(action_int, f"Unknown({action_int})")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"\nStep {i+1}")
        print(f"Actions: {formatted_actions}")
        env.render()
        time.sleep(0.5)  # Pause to see the output
    
    env.close()
    print("\nASCII Rendering Test Complete.")
    
def test_rgb_rendering_matplotlib():
    """Test the RGB array rendering mode using Matplotlib."""
    print("\n--- Testing RGB Array Rendering with Matplotlib ---")
    
    try:
        # Create environment with rgb_array rendering
        env = OvercookedMARL(render_mode="rgb_array", verbose=False)
        
        # Reset environment
        observations, info = env.reset(seed=42)
        
        # Action mapping for display
        action_names = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
            4: "Interact"
        }
        
        # Render and show initial state
        rgb_array = env.render()
        if rgb_array is not None:
            print(f"RGB array shape: {rgb_array.shape}")
            
            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_array)
            plt.title("Initial State")
            plt.axis('off')
            plt.show()
            
            # Create animation of 5 steps
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            frames = []
            
            # Take 5 random steps
            for i in range(5):
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                
                # Format actions for display
                formatted_actions = {}
                for agent, action in actions.items():
                    # Convert numpy int to Python int if needed
                    if hasattr(action, 'item'):
                        action_int = action.item()
                    else:
                        action_int = int(action)
                    
                    formatted_actions[agent] = action_names.get(action_int, f"Unknown({action_int})")
                
                print(f"Step {i+1} - Actions: {formatted_actions}")
                observations, rewards, terminations, truncations, infos = env.step(actions)
                
                rgb_array = env.render()
                frames.append([ax.imshow(rgb_array, animated=True)])
                
                print(f"Step {i+1} rendered")
            
            # Create animation
            ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True)
            plt.title("Random Actions Animation")
            plt.show()
        else:
            print("RGB rendering failed (returned None)")
    except Exception as e:
        print(f"Error in RGB rendering test: {e}")
    finally:
        env.close()
        
    print("\nRGB Rendering Test Complete.")

def test_rgb_rendering_pygame():
    """Test the RGB array rendering mode using Pygame."""
    print("\n--- Testing RGB Array Rendering with Pygame ---")
    
    # Skip this test if pygame is not installed
    if not HAS_PYGAME:
        print("Skipping Pygame RGB rendering test: pygame not installed")
        print("To install pygame, run: pip install pygame")
        return
    
    # Create environment with rgb_array rendering
    env = OvercookedMARL(render_mode="rgb_array", verbose=False)
    
    # Action mapping for display
    action_names = {
        0: "Up",
        1: "Down",
        2: "Left",
        3: "Right",
        4: "Interact"
    }
    
    # Reset environment
    observations, info = env.reset(seed=42)
    
    # Initialize pygame for display
    pygame.init()
    
    # Set up the display
    try:
        # Render initial frame to get its size
        frame = env.render()
        if frame is None:
            print("Error: Failed to render RGB frame")
            env.close()
            pygame.quit()
            return
            
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption("Overcooked MARL - RGB Rendering Test")
    except Exception as e:
        print(f"Error setting up display: {e}")
        env.close()
        pygame.quit()
        return
    
    # Take a few random steps and render each
    try:
        running = True
        step_count = 0
        max_steps = 500  # Increased to allow more visualization time
        clock = pygame.time.Clock()
        
        while running and step_count < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # Press ESC to exit
                        running = False
            
            # Take random actions
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            # Format actions for display
            formatted_actions = {}
            for agent, action in actions.items():
                # Convert numpy int to Python int if needed
                if hasattr(action, 'item'):
                    action_int = action.item()
                else:
                    action_int = int(action)
                
                formatted_actions[agent] = action_names.get(action_int, f"Unknown({action_int})")
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Render and display the frame
            frame = env.render()
            if frame is not None:
                # Convert the NumPy array to a pygame surface
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
                
                # Add informative text at the top of the screen
                font = pygame.font.SysFont(None, 24)
                text_surface = font.render("Press ESC to exit", True, (255, 255, 255), (0, 0, 0))
                screen.blit(text_surface, (10, 10))
                
                pygame.display.flip()
            
            # Display step information
            print(f"\rStep {step_count+1}/{max_steps} - Actions: {formatted_actions}", end="")
            
            step_count += 1
            clock.tick(10)  # 10 FPS
        
        print("\nPygame RGB Rendering Test Complete.")
    except Exception as e:
        print(f"\nError during RGB rendering: {e}")
    finally:
        env.close()
        pygame.quit()

if __name__ == "__main__":
    import argparse
    
    # By default, only ASCII and pygame
    
    # ASCII rendering test (text)
    print("\n=== STARTING ASCII RENDERING TEST ===")
    test_ascii_rendering()
    print("\n=== ASCII RENDERING TEST COMPLETED ===")
    
    # Skip Matplotlib test
    if HAS_MATPLOTLIB:
        print("\nSkipping Matplotlib visualization test (not requested)")
    
    # Pygame rendering test (interactive)
    if HAS_PYGAME:
        print("\n=== STARTING PYGAME RENDERING TEST ===")
        print("Press ESC or close the window to end the test.")
        test_rgb_rendering_pygame()
        print("\n=== PYGAME RENDERING TEST COMPLETED ===")
    else:
        print("\nCould not run Pygame test (pygame not found)")
        print("Run 'pip install pygame' to enable interactive visualization.")