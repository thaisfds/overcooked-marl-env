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
from dataclasses import dataclass
from ..core.items import (
    ItemType, ItemState,
    can_process, is_ready, advance_processing,
    add_ingredient_to_plate, create_plate,
    get_ingredient_display_name
)
from ..core.constants import DEFAULT_CUTS_NEEDED


@dataclass
class RewardValues:
    """
    Configuração centralizada de valores de recompensa.
    
    Filosofia: Recompensar progresso real em direção ao objetivo.
    - Ações básicas (pegar/colocar): recompensa mínima
    - Ações de transformação (processar/combinar): recompensa média
    - Ações de entrega (completar receitas): recompensa alta
    - Penalidades para desencorajar comportamentos improdutivos
    """
    # Ações básicas (sem progresso real em direção ao objetivo)
    pickup: float = 0.1
    place: float = 0.1
    
    # Ações de transformação (progresso real)
    process: float = 3.0      # Cortar ingredientes
    combine: float = 5.0      # Combinar ingredientes com prato
    
    # Recompensas de entrega (objetivo final)
    delivery_base: float = 20.0
    order_bonus_mult: float = 0.5
    
    # Penalidades
    invalid: float = 1.0           # Interação inválida
    useless_cycle: float = 0.5     # Ciclo pegar/colocar repetitivo


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
        self.rewards = rewards or RewardValues()
    
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
        ingredient_dispenser_code = self.item_manager.get_cell_code('I')
        
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
        
        # Ingredient dispenser (from map) - ONLY if not carrying anything
        # If carrying something, treat as counter to allow placing items
        elif tile_pos in game_state["dispensers"] and not carrying_item:
            return self._handle_ingredient_dispenser(
                agent_id, tile_pos, carrying_item,
                game_state, verbose
            )
        
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
        # Includes: '#' (counter), 'I' (ingredient dispenser when carrying items)
        elif station_type in [counter_code, ingredient_dispenser_code]:
            return self._handle_counter(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )
        
        # Invalid interaction
        if verbose:
            print(f"  Invalid interaction location")
        return -self.rewards.invalid, 0.0
    
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
            
            if verbose:
                print(f"  Placed {carrying_item['name']} on chopping board")
            return self.rewards.place, 0.0
        
        # Scenario 2: Chop ingredient on chopping board
        elif not carrying_item and ground_item and can_process(ground_item):
            updated_item = advance_processing(ground_item)
            self._update_item(ground_instance_id, updated_item, game_state)
            
            if verbose:
                cuts_done = updated_item.get("cuts_done", 0)
                cuts_needed = updated_item.get("cuts_needed", DEFAULT_CUTS_NEEDED)
                print(f"  Cutting... {cuts_done}/{cuts_needed} cuts")
                
                if is_ready(updated_item):
                    print(f"  Item processed and ready to use!")
            
            return self.rewards.process, 0.0
        
        # Scenario 3: Pick up processed ingredient from chopping board
        elif not carrying_item and ground_item and is_ready(ground_item):
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]
            
            if verbose:
                print(f"  Picked up {ground_item['name']} from chopping board")
            return self.rewards.pickup, 0.0
        
        # Invalid chopping board interaction
        if verbose:
            print(f"  Invalid chopping board interaction")
        return -self.rewards.invalid, 0.0
    
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
            
            if verbose:
                print(f"  Picked up an empty plate from dispenser (instance: {plate_instance})")
            return self.rewards.pickup, 0.0
        
        # Scenario 2: Pick up plate and combine with READY ingredient
        elif carrying_item and is_ready(carrying_item):
            plate = create_plate()
            plate = add_ingredient_to_plate(plate, carrying_item["name"])
            plate_instance = self._create_item_instance(plate, game_state)
            
            # Delete the ingredient
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            # Now carry the plate
            game_state["agent_holding_instance"][agent_id] = plate_instance
            
            if verbose:
                print(f"  Picked up plate and combined with {carrying_item['name']} (instance: {plate_instance})")
            return self.rewards.pickup + self.rewards.combine, 0.0
        
        # Invalid plate dispenser interaction
        if verbose:
            print(f"  Cannot get plate: carrying non-ready ingredient")
        return -self.rewards.invalid, 0.0
    
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
            if verbose:
                print(f"  Cannot pick up ingredient: hands full")
            return -self.rewards.invalid, 0.0
        
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
        
        if verbose:
            print(f"  Picked up {ingredient['name']} from dispenser (instance: {item_instance})")
        return self.rewards.pickup, 0.0
    
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
            updated_plate = add_ingredient_to_plate(ground_item, carrying_item["name"])
            
            # Delete ingredient
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            # Update plate on counter
            self._update_item(ground_instance_id, updated_plate, game_state)
            
            if verbose:
                print(f"  Combined {carrying_item['name']} with plate on counter")
            return self.rewards.combine, 0.0
        
        # Scenario 2: Carrying plate + READY item on counter → pick up and combine
        elif is_carrying_dish and is_ground_ready:
            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"])
            self._update_item(carrying_instance_id, updated_plate, game_state)
            
            # Delete ingredient from counter
            self._delete_item_instance(ground_instance_id, game_state)
            del game_state["ground_instances"][(ix, iy)]
            
            if verbose:
                print(f"  Combined {ground_item['name']} from counter with plate")
            return self.rewards.combine, 0.0
        
        # Scenario 3: Place item on counter (if empty)
        elif carrying_item and not ground_item:
            game_state["ground_instances"][(ix, iy)] = carrying_instance_id
            del game_state["agent_holding_instance"][agent_id]
            
            reward = self.rewards.place
            if is_useless_cycle:
                reward -= self.rewards.useless_cycle
                if verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter (⚠️ useless cycle: -{self.rewards.useless_cycle})")
            elif verbose:
                print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")
            
            return reward, 0.0
        
        # Scenario 4: Pick up item from counter (if hands empty)
        elif not carrying_item and ground_item:
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]
            
            reward = self.rewards.pickup
            if is_useless_cycle:
                reward -= self.rewards.useless_cycle
                if verbose:
                    print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter (⚠️ useless cycle: -{self.rewards.useless_cycle})")
            elif verbose:
                print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter")
            
            return reward, 0.0
        
        # Invalid counter interaction
        if verbose:
            print(f"  Invalid counter interaction")
        return -self.rewards.invalid, 0.0
    
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
        """
        if carrying_item:
            # Delete the item completely
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            if verbose:
                print(f"  Discarded {get_ingredient_display_name(carrying_item)} in trash")
            return self.rewards.place, 0.0  # Small reward for cleaning up
        
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
        """Handle delivery interactions."""
        # Must be carrying a plate
        if not carrying_item or carrying_item.get("type") != ItemType.DISH:
            if verbose:
                print(f"  Cannot deliver: not carrying a plate")
            return -self.rewards.invalid, 0.0
        
        # Plate must have ingredients
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            if verbose:
                print(f"  Cannot deliver: plate is empty")
            return -self.rewards.invalid, 0.0
        
        # Validate recipe
        is_valid, recipe_name, recipe_reward = self.recipe_manager.validate_delivery(ingredients)
        
        if not is_valid:
            if verbose:
                ingredients_str = ", ".join(ingredients)
                print(f"  Invalid recipe! Ingredients: [{ingredients_str}]")
            return -self.rewards.invalid, 0.0
        
        # Check if it's the correct order
        is_correct_order = self._is_correct_order(recipe_name, game_state)
        
        # Register delivery
        game_state["delivered_recipes"].append(recipe_name)
        game_state["dishes_pending"] -= 1
        
        # Delete delivered plate
        self._delete_item_instance(carrying_instance_id, game_state)
        del game_state["agent_holding_instance"][agent_id]
        
        # Calculate rewards
        base_reward = recipe_reward
        order_bonus = recipe_reward * self.rewards.order_bonus_mult if is_correct_order else 0.0
        global_reward_bonus = recipe_reward + order_bonus
        
        if verbose:
            if is_correct_order:
                print(f"  Recipe {recipe_name} delivered in correct order!")
                print(f"  Reward: {base_reward} + Bonus: {order_bonus}")
            else:
                print(f"  Recipe {recipe_name} delivered out of order")
                print(f"  Reward: {base_reward} (no bonus)")
        
        return base_reward, global_reward_bonus
    
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
        return f"InteractionHandler(rewards: pickup={self.PICKUP_REWARD}, combine={self.COMBINE_REWARD})"
