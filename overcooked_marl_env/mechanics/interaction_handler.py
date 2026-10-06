"""
Interaction Handler - Handles all agent-environment interactions.

This module implements the Single Responsibility Principle (SRP) by separating
interaction logic from the main environment class.

Responsibilities:
- Process interact actions (Action 4)
- Handle different types of interactions:
  * Chopping board interactions (place, process, pickup)
  * Dispenser interactions (ingredient, plate)
  * Counter interactions (place, pickup, combine)
  * Delivery interactions (validate and deliver recipes)
  * Trash interactions (discard items)
"""

from typing import Dict, Tuple, Optional, Any
from ..core.items import (
    ItemType, ItemState,
    can_process, is_ready, advance_processing,
    add_ingredient_to_plate, create_plate,
    get_ingredient_display_name
)
from ..core.constants import DEFAULT_CUTS_NEEDED
from .reward_shaper import RewardValues, RewardShaper


class InteractionHandler:
    """
    Handles all interaction logic for agents.
    
    This class is responsible for:
    1. Processing interact actions (Action 4)
    2. Handling chopping board interactions
    3. Handling dispenser interactions
    4. Handling counter/table interactions
    5. Handling delivery interactions
    6. Calculating rewards for interactions
    
    Uses dependency injection to receive game state and managers.
    """
    
    def __init__(self, item_manager, recipe_manager, rewards: Optional[RewardValues] = None):
        """
        Initialize the interaction handler.
        
        Args:
            item_manager: ItemManager instance for creating items and cell codes
            recipe_manager: RecipeManager instance for validating recipes
            rewards: Configuração de recompensas (opcional, usa padrão se None)
        """
        self.item_manager = item_manager
        self.recipe_manager = recipe_manager
        self.reward_shaper = RewardShaper(item_manager, recipe_manager, rewards)
        
    @property
    def rewards(self) -> RewardValues:
        """Centralized rewards properties for backward compatibility."""
        return self.reward_shaper.rewards
    
    def handle_interaction(
        self,
        agent_id: str,
        game_state: Dict,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Handle an agent's interaction with the environment.
        
        Args:
            agent_id: ID of the agent performing the interaction
            game_state: Current game state dictionary
            verbose: If True, print debug information
        
        Returns:
            Tuple of (local_reward, global_reward_bonus)
            - local_reward: Reward for the specific agent
            - global_reward_bonus: Bonus distributed to all agents
        """
        # Get tile in front of agent
        tile_pos = self._get_tile_in_front(agent_id, game_state)
        ix, iy = tile_pos
        
        # Get what agent is carrying
        carrying_instance_id = game_state["agent_holding_instance"].get(agent_id, None)
        carrying_item = self._get_item(carrying_instance_id, game_state) if carrying_instance_id is not None else None
        
        # Get what's on the ground
        ground_instance_id = game_state["ground_instances"].get((ix, iy), None)
        ground_item = self._get_item(ground_instance_id, game_state) if ground_instance_id is not None else None
        
        # Get station type
        station_type = game_state["map_static"][iy, ix]
        
        # Debug output
        if verbose:
            self._print_interaction_info(
                agent_id, tile_pos, station_type,
                carrying_item, carrying_instance_id,
                ground_item, ground_instance_id
            )
        
        # Route to appropriate handler based on station type
        chopping_code = self.item_manager.get_cell_code('C')
        plate_code = self.item_manager.get_cell_code('P')
        delivery_code = self.item_manager.get_cell_code('D')
        trash_code = self.item_manager.get_cell_code('T')
        counter_code = self.item_manager.get_cell_code('#')
        
        # Chopping board
        if station_type == chopping_code:
            return self._handle_chopping_board(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )
        
        # Plate dispenser
        elif station_type == plate_code:
            return self._handle_plate_dispenser(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
        
        # Ingredient dispenser (from map) - ONLY pickup, NEVER place
        elif tile_pos in game_state["dispensers"]:
            if carrying_item:
                if verbose:
                    print(f"  Cannot place items on ingredient dispenser")
                return 0.0, 0.0  # SEM PENALIDADE
            elif not ground_item:
                return self._handle_ingredient_dispenser(
                    agent_id, tile_pos, carrying_item,
                    game_state, verbose
                )
            else:
                if verbose:
                    print(f"  Invalid state: item on dispenser ground")
                return 0.0, 0.0
        
        # Delivery station
        elif station_type == delivery_code:
            return self._handle_delivery(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
        
        # Trash station - delete items
        elif station_type == trash_code:
            return self._handle_trash(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
        
        # Counter/table (for placing/picking/combining items)
        elif station_type == counter_code:
            return self._handle_counter(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )
        
        # Invalid interaction
        game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
            ('interact_invalid', tile_pos)
        )
        if len(game_state["action_history"][agent_id]) > 10:
            game_state["action_history"][agent_id].pop(0)
            
        repetition_penalty = 0.0
        if self.reward_shaper._check_action_repetition(agent_id, 'interact_invalid', game_state):
            repetition_penalty = self.rewards.repetition_penalty
            
        if verbose:
            print(f"  Invalid interaction location (penalty: {self.rewards.invalid + repetition_penalty:.2f})")
        return self.rewards.invalid + repetition_penalty, 0.0
    
    def _handle_chopping_board(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        ground_item: Optional[Dict],
        ground_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        """Handle interactions with chopping board."""
        ix, iy = tile_pos
        
        # Scenario 1: Place ingredient on chopping board
        if carrying_item and can_process(carrying_item) and not ground_item:
            game_state["ground_instances"][(ix, iy)] = carrying_instance_id
            del game_state["agent_holding_instance"][agent_id]
            
            # Calculate progress reward
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'place_on_chop', carrying_item, None, game_state
            )
            total_reward = self.rewards.place_on_chop + progress_reward
            if total_reward > 0:
                carrying_item["shaping_reward_accumulated"] = carrying_item.get("shaping_reward_accumulated", 0.0) + total_reward
            
            if verbose:
                print(f"  Placed {carrying_item['name']} on chopping board (+{self.rewards.place_on_chop}" + 
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Scenario 2: Chop ingredient on chopping board
        elif not carrying_item and ground_item and can_process(ground_item):
            # Register action in history for repetition detection
            game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
                ('process', tile_pos)
            )
            # Limit history to 10 actions
            if len(game_state["action_history"][agent_id]) > 10:
                game_state["action_history"][agent_id].pop(0)
            
            updated_item = advance_processing(ground_item)
            
            # Check for repetition penalty
            repetition_penalty = 0.0
            if self.reward_shaper._check_action_repetition(agent_id, 'process', game_state):
                repetition_penalty = self.rewards.repetition_penalty
                if verbose:
                    print(f"  WARNING: Repetitive processing detected! Penalty: {repetition_penalty}")
            
            # Check for accumulation penalty (many processed ingredients on ground)
            accumulation_penalty = 0.0
            processed_count = self.reward_shaper._count_processed_ingredients_on_ground(game_state)
            if processed_count >= self.rewards.accumulation_threshold:
                excess = processed_count - self.rewards.accumulation_threshold + 1
                accumulation_penalty = self.rewards.accumulation_penalty * excess
                if verbose:
                    print(f"  WARNING: Too many processed ingredients on ground ({processed_count})! Penalty: {accumulation_penalty}")
            
            cut_reward = self.rewards.process if self.rewards.process != 1.0 else self.rewards.chop_cut
            milestone_reward = self.rewards.chop_completed if is_ready(updated_item) else 0.0
            total_reward = cut_reward + milestone_reward + repetition_penalty + accumulation_penalty
            
            reward_gained = max(0.0, cut_reward + milestone_reward)
            updated_item["shaping_reward_accumulated"] = updated_item.get("shaping_reward_accumulated", 0.0) + reward_gained
            self._update_item(ground_instance_id, updated_item, game_state)
            
            if verbose:
                cuts_done = updated_item.get("cuts_completed", updated_item.get("cuts_done", 0))
                cuts_needed = updated_item.get("cuts_needed", 3)
                print(f"  Cutting... {cuts_done}/{cuts_needed} cuts (+{cut_reward:.1f}" +
                      (f" +{milestone_reward:.1f} complete milestone" if milestone_reward > 0 else "") + ")")
                
                if is_ready(updated_item):
                    print(f"  Item processed and ready to use!")
            
            return total_reward, 0.0
        
        # Scenario 3: Pick up processed ingredient from chopping board
        elif not carrying_item and ground_item and is_ready(ground_item):
            # Calculate progress reward BEFORE moving item (use ground_item)
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'pickup_ready', ground_item, None, game_state
            )
            
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]
            
            total_reward = self.rewards.pickup + progress_reward
            if total_reward > 0:
                ground_item["shaping_reward_accumulated"] = ground_item.get("shaping_reward_accumulated", 0.0) + total_reward
            
            if verbose:
                print(f"  Picked up {ground_item['name']} from chopping board (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Scenario 4: Carrying plate + ready ingredient on chopping board -> combine directly onto plate!
        elif carrying_item and carrying_item.get("type") == ItemType.DISH and ground_item and is_ready(ground_item):
            if ground_item["name"] in carrying_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {ground_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"], ground_item)

            
            # Delete ingredient from chopping board
            self._delete_item_instance(ground_instance_id, game_state)
            del game_state["ground_instances"][(ix, iy)]
            
            # Calculate progress reward
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(updated_plate, game_state)
            
            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            if total_reward > 0:
                updated_plate["shaping_reward_accumulated"] = updated_plate.get("shaping_reward_accumulated", 0.0) + total_reward
            self._update_item(carrying_instance_id, updated_plate, game_state)
            
            if verbose:
                print(f"  Combined {ground_item['name']} from chopping board with plate (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Invalid chopping board interaction
        game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
            ('interact_invalid', tile_pos)
        )
        if len(game_state["action_history"][agent_id]) > 10:
            game_state["action_history"][agent_id].pop(0)

        repetition_penalty = 0.0
        if self.reward_shaper._check_action_repetition(agent_id, 'interact_invalid', game_state):
            repetition_penalty = self.rewards.repetition_penalty

        if verbose:
            print(f"  Invalid chopping board interaction (penalty: {self.rewards.invalid + repetition_penalty:.2f})")
        return self.rewards.invalid + repetition_penalty, 0.0
    
    def _handle_plate_dispenser(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        """Handle interactions with plate dispenser."""
        # Scenario 1: Pick up empty plate
        if not carrying_item:
            plate = create_plate()
            plate_instance = self._create_item_instance(plate, game_state)
            game_state["agent_holding_instance"][agent_id] = plate_instance
            
            # Check if there are ready ingredients on ground nearby (ready to combine)
            ready_ingredient_nearby = self.reward_shaper._check_ready_ingredient_nearby(agent_id, game_state)
            progress_reward = 0.0
            if ready_ingredient_nearby:
                progress_reward = self.rewards.progress_ready_to_combine
                if verbose:
                    print(f"  Ready ingredient nearby - ready to combine!")
            
            total_reward = self.rewards.pickup + progress_reward
            if total_reward > 0:
                plate["shaping_reward_accumulated"] = plate.get("shaping_reward_accumulated", 0.0) + total_reward
                self._update_item(plate_instance, plate, game_state)
            
            if verbose:
                print(f"  Picked up an empty plate from dispenser (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} ready to combine)" if progress_reward > 0 else ""))
            return total_reward, 0.0
        
        # Scenario 2: Pick up plate and combine with READY ingredient
        elif carrying_item and is_ready(carrying_item):
            plate = create_plate()
            plate = add_ingredient_to_plate(plate, carrying_item["name"], carrying_item)
            plate_instance = self._create_item_instance(plate, game_state)
            
            # Delete the ingredient
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            # Now carry the plate
            game_state["agent_holding_instance"][agent_id] = plate_instance
            
            # Calculate progress reward for combining
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, plate, game_state
            )
            
            # Recompensa alta por pegar prato quando já tem ingrediente pronto
            pickup_plate_with_ready_reward = self.rewards.progress_pickup_plate_with_ready
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(plate, game_state)
            
            total_reward = (self.rewards.pickup + self.rewards.combine + 
                          progress_reward + pickup_plate_with_ready_reward + 
                          partial_plate_reward)
            
            if total_reward > 0:
                plate["shaping_reward_accumulated"] = plate.get("shaping_reward_accumulated", 0.0) + total_reward
                self._update_item(plate_instance, plate, game_state)
            
            if verbose:
                print(f"  Picked up plate and combined with {carrying_item['name']} (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      f" +{pickup_plate_with_ready_reward:.1f} pickup_plate_with_ready" +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Invalid plate dispenser interaction
        if verbose:
            print(f"  Cannot get plate: carrying non-ready ingredient")
        return self.rewards.invalid, 0.0
    
    def _handle_ingredient_dispenser(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        """Handle interactions with ingredient dispensers."""
        # Can only pick up if not carrying anything
        if carrying_item:
            game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
                ('interact_invalid', tile_pos)
            )
            if len(game_state["action_history"][agent_id]) > 10:
                game_state["action_history"][agent_id].pop(0)

            repetition_penalty = 0.0
            if self.reward_shaper._check_action_repetition(agent_id, 'interact_invalid', game_state):
                repetition_penalty = self.rewards.repetition_penalty

            if verbose:
                print(f"  Cannot pick up ingredient: hands full (penalty: {self.rewards.invalid + repetition_penalty:.2f})")
            return self.rewards.invalid + repetition_penalty, 0.0
        
        dispenser_data = game_state["dispensers"][tile_pos]
        
        # Support both name (string) and numeric ID (legacy)
        if isinstance(dispenser_data, str):
            ingredient = self.item_manager.create_ingredient_by_name(dispenser_data)
        else:
            # Legacy: numeric ID
            ingredient_map = {
                5: "tomato",
                6: "lettuce",
                7: "cucumber",
            }
            ingredient_name = ingredient_map.get(dispenser_data, "tomato")
            ingredient = self.item_manager.create_ingredient_by_name(ingredient_name)
        
        item_instance = self._create_item_instance(ingredient, game_state)
        game_state["agent_holding_instance"][agent_id] = item_instance
        
        # Calculate progress reward
        progress_reward = self.reward_shaper._calculate_progress_reward(
            'pickup_ingredient', ingredient, None, game_state
        )
        total_reward = self.rewards.pickup + progress_reward
        if total_reward > 0:
            ingredient["shaping_reward_accumulated"] = ingredient.get("shaping_reward_accumulated", 0.0) + total_reward
            self._update_item(item_instance, ingredient, game_state)
        
        if verbose:
            print(f"  Picked up {ingredient['name']} from dispenser (instance: {item_instance})" +
                  (f" (+{progress_reward:.1f} progress)" if progress_reward > 0 else ""))
        return total_reward, 0.0
    
    def _handle_counter(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        ground_item: Optional[Dict],
        ground_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        """Handle interactions with counters (place, pickup, combine)."""
        ix, iy = tile_pos
        
        is_carrying_dish = carrying_item and carrying_item.get("type") == ItemType.DISH
        is_carrying_ready = carrying_item and is_ready(carrying_item)
        is_ground_dish = ground_item and ground_item.get("type") == ItemType.DISH
        is_ground_ready = ground_item and is_ready(ground_item)
        
        # Detecta ciclo inútil (pegar/colocar repetitivamente no mesmo lugar)
        is_useless_cycle = False
        history = game_state.get("action_history", {}).get(agent_id, [])
        
        if len(history) >= 3:
            # Últimas 3 ações
            recent = history[-3:]
            actions = [a[0] for a in recent]
            positions = [a[1] for a in recent]
            
            # Verifica se alterna pickup/place, mesma posição, apenas essas ações
            is_alternating = all(actions[i] != actions[i+1] for i in range(len(actions)-1))
            same_pos = len(set(positions)) == 1 and positions[0] == tile_pos
            only_pickup_place = all(a in ["pickup", "place"] for a in actions)
            
            is_useless_cycle = is_alternating and same_pos and only_pickup_place
        
        # Registra ação atual no histórico
        action_type = "place" if carrying_item else "pickup"
        game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
            (action_type, tile_pos)
        )
        # Limita histórico a 10 ações
        if len(game_state["action_history"][agent_id]) > 10:
            game_state["action_history"][agent_id].pop(0)
        
        # Scenario 1: Carrying READY item + plate on counter → combine
        if is_carrying_ready and is_ground_dish:
            if carrying_item["name"] in ground_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {carrying_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(ground_item, carrying_item["name"], carrying_item)
            
            # Delete ingredient
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            # Calculate progress reward
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(updated_plate, game_state)
            
            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            if total_reward > 0:
                updated_plate["shaping_reward_accumulated"] = updated_plate.get("shaping_reward_accumulated", 0.0) + total_reward
            
            # Update plate on counter
            self._update_item(ground_instance_id, updated_plate, game_state)
            
            if verbose:
                print(f"  Combined {carrying_item['name']} with plate on counter (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Scenario 2: Carrying plate + READY item on counter → pick up and combine
        elif is_carrying_dish and is_ground_ready:
            if ground_item["name"] in carrying_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {ground_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"], ground_item)

            
            # Delete ingredient from counter
            self._delete_item_instance(ground_instance_id, game_state)
            del game_state["ground_instances"][(ix, iy)]
            
            # Calculate progress reward
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(updated_plate, game_state)
            
            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            if total_reward > 0:
                updated_plate["shaping_reward_accumulated"] = updated_plate.get("shaping_reward_accumulated", 0.0) + total_reward
            
            self._update_item(carrying_instance_id, updated_plate, game_state)
            
            if verbose:
                print(f"  Combined {ground_item['name']} from counter with plate (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Scenario 3: Place item on counter (if empty)
        elif carrying_item and not ground_item:
            game_state["ground_instances"][(ix, iy)] = carrying_instance_id
            del game_state["agent_holding_instance"][agent_id]
            
            # Conta quantos itens já estão nas bancadas (Counter = '#' = code 1)
            counter_code = self.item_manager.get_cell_code('#')
            total_counter_items = sum(1 for pos, _ in game_state["ground_instances"].items() 
                                     if game_state["map_static"][pos[1], pos[0]] == counter_code)
            
            reward = self.rewards.place
            if is_useless_cycle:
                reward += self.rewards.useless_cycle_penalty
                if verbose:
                    print(f"  WARNING: Useless pickup/place cycle detected! Penalty: {self.rewards.useless_cycle_penalty}")
            
            # Aplica penalidade progressiva se houver mais de 3 itens em bancadas
            if total_counter_items > 3:
                excess_items = total_counter_items - 3
                clutter_penalty = self.rewards.counter_clutter_penalty * (excess_items ** 1.5)
                reward += clutter_penalty
                if verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")
                    print(f"  Counter clutter penalty: {clutter_penalty:.2f} ({total_counter_items} items on counters)")
            else:
                if verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")
            
            return reward, 0.0
        
        # Scenario 4: Pick up item from counter (if hands empty)
        elif not carrying_item and ground_item:
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]
            
            # Calculate progress reward based on what was picked up
            progress_reward = 0.0
            if is_ready(ground_item):
                # Picking up ready ingredient
                progress_reward = self.reward_shaper._calculate_progress_reward(
                    'pickup_ready', ground_item, None, game_state
                )
            elif ground_item.get("type") == ItemType.DISH:
                # Picking up plate - check if it's complete
                progress_reward = self.reward_shaper._calculate_progress_reward(
                    'pickup_plate_complete', ground_item, None, game_state
                )
            
            total_reward = self.rewards.pickup + progress_reward
            if total_reward > 0:
                ground_item["shaping_reward_accumulated"] = ground_item.get("shaping_reward_accumulated", 0.0) + total_reward
                self._update_item(ground_instance_id, ground_item, game_state)
            
            if verbose:
                print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            
            return total_reward, 0.0
        
        # Invalid counter interaction
        if verbose:
            print(f"  Invalid counter interaction (penalty: {self.rewards.invalid})")
        return self.rewards.invalid, 0.0
    
    def _handle_trash(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        """
        Handle interactions with trash station.
        
        Trash deletes items completely - they cease to exist.
        Applies exact reward clawback of all accumulated shaping rewards plus base trash penalty.
        """
        if carrying_item:
            shaping_accumulated = carrying_item.get("shaping_reward_accumulated", 0.0)
            # Penalty is full clawback of all accumulated shaping rewards + base trash penalty
            penalty = -shaping_accumulated + self.rewards.trash_penalty

            # Delete the item completely
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            if verbose:
                print(f"  Discarded {get_ingredient_display_name(carrying_item)} in trash (Estorno: -{shaping_accumulated:.1f}, Taxa: {self.rewards.trash_penalty:.1f} => Total: {penalty:.1f})")
            return penalty, 0.0  # PENALIDADE por desperdiçar item
        
        if verbose:
            print(f"  Cannot use trash: not carrying anything")
        return 0.0, 0.0  # No penalty, just no action
    
    def _handle_delivery(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        invalid_pen = self.rewards.invalid_delivery_penalty
        
        # Must be carrying a plate
        if not carrying_item or carrying_item.get("type") != ItemType.DISH:
            if verbose:
                print(f"  Cannot deliver: not carrying a plate")
            return invalid_pen, 0.0
        
        # Plate must have ingredients
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            if verbose:
                print(f"  Cannot deliver: plate is empty")
            return invalid_pen, 0.0
        
        # Validate recipe
        is_valid, recipe_name, recipe_reward = self.recipe_manager.validate_delivery(ingredients)
        
        if not is_valid:
            if verbose:
                ingredients_str = ", ".join(ingredients)
                print(f"  Invalid recipe! Ingredients: [{ingredients_str}] (penalty: {invalid_pen})")
            return invalid_pen, 0.0
        
        # Get remaining pending orders in HUD queue
        recipe_order = game_state.get("recipe_order", [])
        pending_orders = list(recipe_order)
        for delivered in game_state.get("delivered_recipes", []):
            if delivered in pending_orders:
                pending_orders.remove(delivered)
        
        # Must match a recipe currently requested in pending HUD orders
        if recipe_name not in pending_orders:
            unrequested_penalty = -abs(recipe_reward)
            if verbose:
                print(f"  Recipe {recipe_name} is not in pending HUD orders! Wiping dish points (penalty: {unrequested_penalty:.1f})")
            
            # Delete unrequested plate to clear hands, but do NOT count as successful delivery
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            return unrequested_penalty, unrequested_penalty
            
        # Check if it's strictly the top order in HUD queue (for order bonus)
        is_correct_order = (len(pending_orders) > 0 and recipe_name == pending_orders[0])
        
        # Register delivery
        game_state["delivered_recipes"].append(recipe_name)
        game_state["dishes_pending"] -= 1
        
        # Delete delivered plate
        self._delete_item_instance(carrying_instance_id, game_state)
        del game_state["agent_holding_instance"][agent_id]
        
        # Calculate rewards
        base_reward = recipe_reward
        order_bonus = (recipe_reward * self.rewards.order_bonus_mult) if is_correct_order else 0.0
        
        # Sequence complete bonus
        sequence_bonus = self.rewards.sequence_complete_bonus
        
        # Delivery multipliers
        num_deliveries = len(game_state.get("delivered_recipes", []))
        if num_deliveries == 1:
            delivery_multiplier = 1.0
        elif num_deliveries == 2:
            delivery_multiplier = self.rewards.delivery_multiplier_2nd
        elif num_deliveries == 3:
            delivery_multiplier = self.rewards.delivery_multiplier_3rd
        else:
            delivery_multiplier = self.rewards.delivery_multiplier_4th_plus
        
        delivery_bonus = base_reward * (delivery_multiplier - 1.0)
        
        # Check if episode is now complete (all dishes delivered)
        episode_complete_bonus = 0.0
        if game_state["dishes_pending"] <= 0:
            episode_complete_bonus = self.rewards.episode_complete_bonus
            if verbose:
                print(f"  ALL DISHES DELIVERED! Episode Victory Bonus: +{episode_complete_bonus}")
        
        total_reward = base_reward + order_bonus + sequence_bonus + delivery_bonus + episode_complete_bonus
        global_reward_bonus = recipe_reward + order_bonus + delivery_bonus + episode_complete_bonus
        
        if verbose:
            if is_correct_order:
                print(f"  Recipe {recipe_name} delivered in correct order! (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Order Bonus: {order_bonus} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f})" +
                      (f" + Victory Bonus: {episode_complete_bonus}" if episode_complete_bonus > 0 else ""))
            else:
                print(f"  Recipe {recipe_name} delivered out of order (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f}) (no order bonus)" +
                      (f" + Victory Bonus: {episode_complete_bonus}" if episode_complete_bonus > 0 else ""))
        
        return total_reward, global_reward_bonus
    
    def _is_correct_order(self, recipe_name: str, game_state: Dict) -> bool:
        """Check if delivered recipe is in correct order."""
        recipe_order = game_state.get("recipe_order", [])
        if not recipe_order:
            return False
        
        # Calculate pending orders
        pending_orders = list(recipe_order)
        for delivered_recipe in game_state.get("delivered_recipes", []):
            if delivered_recipe in pending_orders:
                pending_orders.remove(delivered_recipe)
        
        # Check if it's the first pending recipe
        return len(pending_orders) > 0 and recipe_name == pending_orders[0]
    

    
    # Helper methods for item management
    
    def _get_tile_in_front(self, agent_id: str, game_state: Dict) -> Tuple[int, int]:
        """Get the tile position in front of the agent."""
        agent_pos = game_state["agent_pos"][agent_id]
        x, y = agent_pos
        
        # Get agent orientation
        dx, dy = game_state["agent_orientation"][agent_id]
        
        # Calculate position in front
        new_x, new_y = x + dx, y + dy
        
        # Check if within bounds
        map_height, map_width = game_state["map_static"].shape
        if 0 <= new_x < map_width and 0 <= new_y < map_height:
            return (new_x, new_y)
        
        # If out of bounds, return current position
        return agent_pos
    
    def _get_item(self, item_instance_id: Optional[int], game_state: Dict) -> Optional[Dict]:
        """Get item dictionary from instance ID."""
        if item_instance_id is None:
            return None
        return game_state["item_instances"].get(item_instance_id)
    
    def _create_item_instance(self, item_dict: Dict, game_state: Dict) -> int:
        """Create a new item instance and return its ID."""
        item_id = game_state["next_item_id"]
        game_state["next_item_id"] += 1
        game_state["item_instances"][item_id] = item_dict.copy()
        return item_id
    
    def _update_item(self, item_instance_id: int, new_item_dict: Dict, game_state: Dict):
        """Update an item instance."""
        if item_instance_id in game_state["item_instances"]:
            game_state["item_instances"][item_instance_id] = new_item_dict.copy()
    
    def _delete_item_instance(self, item_instance_id: int, game_state: Dict):
        """Delete an item instance."""
        if item_instance_id in game_state["item_instances"]:
            del game_state["item_instances"][item_instance_id]
    
    def _print_interaction_info(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        station_type: int,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        ground_item: Optional[Dict],
        ground_instance_id: Optional[int]
    ):
        """Print debug information about the interaction."""
        ix, iy = tile_pos
        print(f"Agent {agent_id} interacting at position ({int(ix)}, {int(iy)})")
        print(f"  Station type: {station_type}")
        
        if carrying_item:
            print(f"  Carrying: {get_ingredient_display_name(carrying_item)} (instance: {carrying_instance_id})")
        else:
            print(f"  Carrying: nothing")
        
        if ground_item:
            print(f"  Ground: {get_ingredient_display_name(ground_item)} (instance: {ground_instance_id})")
        else:
            print(f"  Ground: nothing")
    
    def __repr__(self) -> str:
        """String representation of InteractionHandler."""
        return f"InteractionHandler(rewards: pickup={self.rewards.pickup}, combine={self.rewards.combine})"
