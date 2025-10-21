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

## Phase 6: Extensions (Future)
- [ ] Advanced Features
    - [ ] Action masking for valid moves
    - [ ] Configurable difficulty levels
    - [ ] Dynamic recipe generation
    - [ ] More map elements and interactions
- [ ] Documentation
    - [ ] API documentation
    - [ ] Usage examples
    - [ ] Custom map creation guide
- [ ] Training Examples
    - [ ] Simple agent implementation
    - [ ] Cooperative policy training
