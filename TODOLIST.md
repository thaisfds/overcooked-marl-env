# TODO LIST

## Phase 1: Foundation (Completed)
- [x] Setup and Initial Structure
    - [x] Create project repository
    - [x] Setup virtual environment
    - [x] Install dependencies (gymnasium, numpy, pettingzoo)

## Phase 2: Design (Completed)
- [x] MARL (Multi-Agent) Environment Design
    - [x] Define Agents (Variable agent system using PettingZoo)
    - [x] Define Observation Spaces (3D Box with state layers)
    - [x] Define Action Spaces (Discrete: 4 movement + 1 interact)
    - [x] Reward Design (Hybrid: individual + global rewards)
    - [x] Termination Design (Time limit and recipe completion)

## Phase 3: Core Implementation (Completed)
- [x] Environment Implementation
    - [x] Main Class (OvercookedMARL inheriting from PettingZoo.ParallelEnv)
    - [x] __init__ (Agent setup, spaces, map loading)
    - [x] _get_obs (Multi-channel observation generation)
    - [x] _get_info (Recipe info and agent status)
    - [x] reset() (Map reinitialization)
    - [x] step(actions) (Movement, interaction, rewards)
    - [x] Required ParallelEnv methods (observe, state, action_space, observation_space)

## Phase 4: Logic Systems (Completed)
- [x] Map Loading and Processing
    - [x] Parse map files with grid, agent positions, dispensers
    - [x] Support for multiple recipes
- [x] Recipe Management
    - [x] Recipe definitions and ingredient requirements
    - [x] Order tracking and validation
    - [x] Recipe-specific rewards
- [x] Interaction System
    - [x] Item collection from dispensers
    - [x] Processing interactions (cutting, cooking)
    - [x] Delivery and scoring

## Phase 5: Testing and Refinement (Completed)
- [x] Basic Testing
    - [x] Sanity checks for environment structure
    - [x] PettingZoo API compliance tests
    - [x] Random action execution tests
- [x] Code Quality
    - [x] Standardize comments in English
    - [x] Improve error handling
- [x] Visualization and Debugging
    - [x] Implement render() method
    - [x] ASCII rendering for "human" mode
    - [x] RGB array rendering for "rgb_array" mode (using pygame)
    - [x] Add visualization dependencies to setup.py

## Phase 6: Overcooked Feature Parity - HIGH PRIORITY 🔥
**Mecânicas centrais ausentes do jogo original que adicionam profundidade estratégica significativa**

- [ ] **Dirty Dishes & Washing System** ⭐ MAIS IMPORTANTE
    - [ ] Implement dirty plate mechanics (plates become dirty after delivery)
    - [ ] Add sink station ('S') for washing dishes
    - [ ] Limit clean plate stack (e.g., 3 plates maximum)
    - [ ] Add washing time/interaction requirement
    - [ ] Update observation space to track dirty vs clean plates
    - [ ] Add rewards for washing (+0.5) and penalties for no clean plates available

- [ ] **Cooking/Frying System** 🍳
    - [ ] Add stove/oven station ('O' or 'F')
    - [ ] Implement cooking state for ingredients (RAW → COOKING → COOKED)
    - [ ] Add timer system for cooking progress
    - [ ] Update item states: add COOKING and COOKED states
    - [ ] Modify recipes to require cooked ingredients (e.g., meat must be cooked)
    - [ ] Add rewards for proper cooking (+0.5 per cooked item)

- [ ] **Burning Mechanics** 🔥
    - [ ] Add BURNING and BURNT item states
    - [ ] Implement timer for overcooking (e.g., 10 steps after COOKED → BURNT)
    - [ ] Make burnt items unusable (cannot add to plates)
    - [ ] Add visual indicators for burning (in both render modes)
    - [ ] Penalty for burning food (-1.0)
    - [ ] Require trash station to dispose burnt items

## Phase 7: Overcooked Feature Parity - MEDIUM PRIORITY ⚡
**Mecânicas que aumentam dinâmica e profundidade do gameplay**

- [ ] **Order Timers & Dynamic Requests** ⏰
    - [ ] Implement individual timer per recipe order
    - [ ] Add time bonus for early delivery (multiplier based on remaining time)
    - [ ] Add penalty for expired orders (-5.0)
    - [ ] Visual countdown in render mode
    - [ ] Queue system for multiple simultaneous orders
    - [ ] Update observation space to include timer information

- [ ] **Combo/Streak Multiplier System** 📊
    - [ ] Track consecutive successful deliveries
    - [ ] Implement multiplier system (1x → 2x → 3x → 4x)
    - [ ] Add time window for maintaining streak (e.g., 20 steps between deliveries)
    - [ ] Reset multiplier on failed delivery or timeout
    - [ ] Display current multiplier in info and render
    - [ ] Bonus rewards scale with multiplier

- [ ] **Functional Trash Station** 🗑️
    - [ ] Make 'T' station functional for discarding items
    - [ ] Allow disposal of burnt food, wrong ingredients, or unwanted items
    - [ ] Clear agent inventory when using trash
    - [ ] Small penalty for wasting food (-0.2 for non-burnt items)
    - [ ] No penalty for disposing burnt items

## Phase 8: Overcooked Feature Parity - LOW PRIORITY 🎨
**Polimento e features avançadas que melhoram fluidez**

- [ ] **Direct Item Passing Between Agents** 🤝
    - [ ] Allow agents to pass items directly without counter
    - [ ] Implement "give" interaction when facing another agent
    - [ ] Add "receive" state for target agent
    - [ ] Reward for successful coordination (+0.3)
    - [ ] Update observation to show when agent can receive

- [ ] **Dynamic Environmental Obstacles** 🌊
    - [ ] Moving platforms (change position periodically)
    - [ ] Teleportation portals (instant transport between points)
    - [ ] Conveyor belts (automatic item/agent movement)
    - [ ] Doors/gates (open/close based on conditions)
    - [ ] Update map parser to support dynamic elements

- [ ] **Enhanced Interaction System** ⚙️
    - [ ] Fix `_get_tile_in_front()` to prioritize orientation over current cell
    - [ ] Separate plate pickup from auto-combine at dispenser
    - [ ] Add "hold" interaction progress bars for visual feedback
    - [ ] Multi-step interactions (hold button for faster processing)

## Phase 9: Extensions (Future)
- [ ] Advanced Features
    - [ ] Action masking for valid moves
    - [ ] Configurable difficulty levels
    - [ ] Dynamic recipe generation
    - [ ] More map elements and interactions
    - [ ] Save/load game states
    - [ ] Replay system for debugging

- [ ] Documentation
    - [x] Environment rules documentation (REGRAS_AMBIENTE.md)
    - [ ] API documentation (Sphinx/MkDocs)
    - [ ] Usage examples and tutorials
    - [ ] Custom map creation guide
    - [ ] Recipe creation guide
    - [ ] Video demonstrations

- [ ] Training Examples
    - [ ] Simple agent implementation (random, heuristic)
    - [ ] Cooperative policy training (PPO, MAPPO)
    - [ ] Curriculum learning examples
    - [ ] Benchmark scenarios and metrics
    - [ ] Pre-trained model checkpoints

## Phase 10: Performance & Quality of Life
- [ ] Performance Optimization
    - [ ] Profile and optimize observation generation
    - [ ] Vectorized environment support
    - [ ] Parallel episode execution
    - [ ] Memory usage optimization

- [ ] Quality of Life
    - [ ] Configuration file support (YAML/JSON)
    - [ ] Logging system with different verbosity levels
    - [ ] Detailed statistics tracking
    - [ ] Debug mode with step-by-step visualization
    - [ ] Unit tests for all major systems
    - [ ] Integration tests for full episodes

---

## 📝 Notes

### Priority Explanation
- **HIGH PRIORITY**: Core Overcooked mechanics that define the game's challenge
- **MEDIUM PRIORITY**: Features that add strategic depth and replayability  
- **LOW PRIORITY**: Polish and advanced features for enhanced experience

### Implementation Order Suggestion
1. Start with **Dirty Dishes** (biggest impact on cooperation)
2. Add **Cooking System** (new station type and mechanics)
3. Implement **Burning** (builds on cooking system)
4. Add **Timers** (increases pressure and dynamics)
5. Implement remaining features based on project goals

### Testing Strategy
- Test each new mechanic in isolation with dedicated test scripts
- Ensure backward compatibility with existing maps
- Create new maps showcasing new mechanics
- Validate that new features don't break existing trained agents