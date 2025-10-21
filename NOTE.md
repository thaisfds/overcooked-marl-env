# Notes

python.exe -m pip install --upgrade pip

py -3.13 -m venv venv

venv\Scripts\activate

pip install gymnasium numpy pettingzoo # Add 'pettingzoo'

pip install -e .

python enviroment_tests/salad_lettuce.py > test_log.txt

deactivate

# Decisions
Summary: gymnasium environment for MARL based on overcooked

### MARL Environment Design
##### Define Agents
Variable Agents - In the implementation of step(), you'll need to manage a dictionary of active agents and return data only for them. PettingZoo is the ideal library for this, as it's built to handle the sequential or parallel nature of games with variable agents/turns.

##### Define Observation Spaces
Observation Spaces (Global Vision) - You'll use a gymnasium.spaces.Box or gymnasium.spaces.Dict for the observation. I recommend a 3D Box (width x height x state layers) where each layer represents the state (e.g., 1st layer: agent location, 2nd layer: tomato location, 3rd layer: cutting station, etc.).

##### Define Action Spaces
Action Spaces (Walk + Interact) - You'll use a gymnasium.spaces.Discrete where the first 4 values are directions and the remaining values are interactions (e.g., 0-3: Walk, 4: Interact).

Collision and Restrictions - The challenge here is Action Masking. In each step(), the agent will try an action, but you'll need to check if it's valid (e.g., no wall, station or agent in the way). If the action is invalid, you can:
1) Not move the agent (stay still), or
2) Apply a small reward penalty.

##### Reward Design
Hybrid Reward - This is excellent. In the step() method, you'll calculate:

Individual Reward (local): If the action was a successful interaction (e.g., picked up an item, placed an item at the station).

Collective Reward (global): If the action resulted in the delivery of a dish. This reward should be added to the individual value and distributed to all agents in the rewards: dict.

##### Termination Design
Time Limit - This translates to the truncated signal. It's Gymnasium's way of saying the episode ended for a reason not directly related to the task's failure or success (e.g., time out).

Delivered Dishes - This translates to the terminated signal. It's the natural conclusion of the task (success or critical failure).

### Environment Implementation


# AI TIPS: MARL Environment Implementation Order

Phase 1: Skeleton and Spaces (The Gymnasium/PettingZoo Structure)

This phase is about defining the "shape" of your environment. If you get this wrong, nothing else will work.

    Create the Base Class: Define the OvercookedMARL class inheriting from ParallelEnv (from PettingZoo).

    Define Static Variables: In __init__, define the list of agent IDs (self.agents) and the map size (self.map_size).

    Define Observation & Action Spaces: Implement self.observation_spaces and self.action_spaces as dictionaries of gymnasium.spaces.

        Observation: Define the exact dimensions (width x height x state layers) of your 3D Box.

        Action: Define the total number of discrete actions (4 movements + 1 interact) in your Discrete(5).

    Implement _initialize_game_state and _get_obs: Create these helper functions. At first, they can return dummy data (np.zeros or spaces.sample()) just to ensure the input/output data format is correct.

    Implement reset(): Implement reset() to initialize the game state and return the first observations and information using your helper methods.

Phase 2: Core Logic (The Game Engine)

This is the most complex phase, where the simulation takes place.

    Implement step() (Input Logic): Set up the step(actions) method to receive the action dictionary and initialize the 5 return dictionaries (obs, rewards, term, trunc, info).

    Movement and Collision Logic: Within the step() loop, implement the agent movement logic, ensuring collision rules (with walls, stations, and other agents) are respected.

    Interaction Logic: Implement the main Overcooked logic: what happens when an agent uses the "Interact" action in different states (pick up item, cut item, combine item).

    Hybrid Reward Logic:

        Calculate small individual rewards (for progression).

        Calculate the large global reward (for dish delivery) and distribute to all agents.

    Termination Logic: Implement the conditions for terminated (dishes delivered) and truncated (time limit), ensuring that the return dictionary __all__ is marked as True when the episode ends.

Phase 3: Finalization and Testing

    Implement render() (Optional but Recommended): Add logic to render the map (simple ASCII if using render_mode="human", or a pixel matrix for render_mode="rgb_array").

    Run check_env: Use the check_env function from Gymnasium (and parallel_api_test from PettingZoo, if desired) to formally validate that you followed the API protocol.

    Run the Sanity Test: Run the enviroment_tests/test.py script to ensure the environment runs 100 steps without crashing, confirming that the game logic is stable.

    Finalize the Setup: Make sure your setup.py and registration in __init__.py are correct so others can use gym.make().

The Golden Rule: Don't move to the next phase until the code for the current phase works perfectly with your test code.


