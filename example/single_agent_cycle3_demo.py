"""
Scripted end-to-end demonstration for Cycle 3:
1 Agent executing all steps to fulfill and deliver all orders on default_map.txt.
"""

from overcooked_marl_env.env import OvercookedMARL

def run_single_agent_demo():
    print("=" * 60)
    print("CYCLE 3 VERIFICATION: Single Agent Delivering All Recipes")
    print("=" * 60)

    env = OvercookedMARL(render_mode=None, num_fixed_agents=1, featurized_obs=True)
    obs, info = env.reset()
    agent = "agent_0"

    total_episode_reward = 0.0

    def step_action(action, desc=""):
        nonlocal total_episode_reward
        o, r, term, trunc, inf = env.step({agent: action})
        total_episode_reward += r[agent]
        if desc:
            print(f"[{env.state['time_steps']:03d}] {desc:<45} | Reward: {r[agent]:+7.2f} | Acc: {total_episode_reward:7.2f}")
        return o, r, term, trunc, inf

    def navigate_to(target_pos, facing_dir):
        # 0: Up (0, -1), 1: Down (0, 1), 2: Left (-1, 0), 3: Right (1, 0)
        curr = env.state["agent_pos"][agent]
        while curr[0] < target_pos[0]:
            step_action(3)
            curr = env.state["agent_pos"][agent]
        while curr[0] > target_pos[0]:
            step_action(2)
            curr = env.state["agent_pos"][agent]
        while curr[1] < target_pos[1]:
            step_action(1)
            curr = env.state["agent_pos"][agent]
        while curr[1] > target_pos[1]:
            step_action(0)
            curr = env.state["agent_pos"][agent]
        
        # Face required direction
        orient_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        step_action(orient_map[facing_dir])

    # Dispensers: Tomato is at (7, 5), Lettuce is at (8, 5)
    # Chopping board C is at (7, 1)
    # Plate dispenser P is at (9, 1)
    # Delivery station D is at (10, 3)

    print("\n--- ORDER 1: SIMPLE_SALAD (Lettuce) ---")
    navigate_to((8, 4), (0, 1))
    step_action(4, "Pick up Lettuce from dispenser (8, 5)")

    navigate_to((7, 2), (0, -1))
    step_action(4, "Place raw Lettuce on Chopping Board (7, 1)")

    step_action(4, "Chop cut 1/3")
    step_action(4, "Chop cut 2/3")
    step_action(4, "Chop cut 3/3 (Completed!)")

    navigate_to((9, 2), (0, -1))
    step_action(4, "Pick up Plate from Dispenser (9, 1)")

    navigate_to((7, 2), (0, -1))
    step_action(4, "Combine Plate with Chopped Lettuce (7, 1)")

    navigate_to((9, 3), (1, 0))
    o, r, term, trunc, inf = step_action(4, "DELIVER ORDER 1: SIMPLE_SALAD at (10, 3)")
    print(f"Delivered recipes: {env.state['delivered_recipes']} | Remaining: {env.state['dishes_pending']}")

    print("\n--- ORDER 2: GARDEN_SALAD (Lettuce + Tomato) ---")
    # 1. Lettuce
    navigate_to((8, 4), (0, 1))
    step_action(4, "Pick up Lettuce from dispenser (8, 5)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Place Lettuce on Chopping Board (7, 1)")
    step_action(4, "Chop cut 1/3")
    step_action(4, "Chop cut 2/3")
    step_action(4, "Chop cut 3/3 (Completed!)")
    navigate_to((9, 2), (0, -1))
    step_action(4, "Pick up Plate from Dispenser (9, 1)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Combine Plate with Chopped Lettuce")
    # Put partial plate on counter (6, 2) facing West
    navigate_to((6, 2), (-1, 0))
    step_action(4, "Store partial plate on counter (5, 2)")

    # 2. Tomato
    navigate_to((7, 4), (0, 1))
    step_action(4, "Pick up Tomato from dispenser (7, 5)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Place Tomato on Chopping Board (7, 1)")
    step_action(4, "Chop cut 1/3")
    step_action(4, "Chop cut 2/3")
    step_action(4, "Chop cut 3/3 (Completed!)")

    # Pick up plate from counter and combine with tomato
    navigate_to((6, 2), (-1, 0))
    step_action(4, "Pick up Plate with Lettuce from counter (5, 2)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Combine Plate with Chopped Tomato (Garden Salad Complete!)")

    navigate_to((9, 3), (1, 0))
    o, r, term, trunc, inf = step_action(4, "DELIVER ORDER 2: GARDEN_SALAD at (10, 3)")
    print(f"Delivered recipes: {env.state['delivered_recipes']} | Remaining: {env.state['dishes_pending']}")

    print("\n--- ORDER 3: SIMPLE_SALAD (Lettuce) ---")
    navigate_to((8, 4), (0, 1))
    step_action(4, "Pick up Lettuce from dispenser (8, 5)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Place Lettuce on Chopping Board (7, 1)")
    step_action(4, "Chop cut 1/3")
    step_action(4, "Chop cut 2/3")
    step_action(4, "Chop cut 3/3 (Completed!)")
    navigate_to((9, 2), (0, -1))
    step_action(4, "Pick up Plate from Dispenser (9, 1)")
    navigate_to((7, 2), (0, -1))
    step_action(4, "Combine Plate with Chopped Lettuce")
    navigate_to((9, 3), (1, 0))
    o, r, term, trunc, inf = step_action(4, "DELIVER ORDER 3: SIMPLE_SALAD (FINAL ORDER!)")
    print(f"Delivered recipes: {env.state['delivered_recipes']} | Remaining: {env.state['dishes_pending']}")

    print("\n" + "=" * 60)
    print(f"EPISODE RESULT:")
    print(f"  Terminated (Victory): {term[agent]}")
    print(f"  Delivered recipes:    {env.state['delivered_recipes']}")
    print(f"  Dishes remaining:     {env.state['dishes_pending']}")
    print(f"  Total steps:          {env.state['time_steps']}")
    print(f"  Total reward earned:  {total_episode_reward:.2f}")
    print("=" * 60)

    assert term[agent], "Episode should be terminated with victory!"
    assert env.state['dishes_pending'] == 0, "All dishes should be delivered!"
    assert len(env.state['delivered_recipes']) == 3, "Exactly 3 recipes should be delivered!"
    print("SUCCESS: 1 agent successfully completed and delivered all orders!")

if __name__ == "__main__":
    run_single_agent_demo()
