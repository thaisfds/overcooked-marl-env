"""
Interaction Handler - Handles all agent-environment interactions.

This module coordinates interactions by delegating station-specific logic to modular
StationHandlers using the Strategy Pattern.

Supported Stations:
- Chopping board ('C') -> ChoppingBoardHandler
- Plate dispenser ('P') -> PlateDispenserHandler
- Ingredient dispenser ('I', map dispensers) -> IngredientDispenserHandler
- Counter/table ('#') -> CounterHandler
- Trash station ('T') -> TrashHandler
- Delivery station ('D') -> DeliveryHandler
"""

from typing import Dict, Tuple, Optional, Any
from ..core.items import (
    ItemType, ItemState,
    can_process, is_ready, is_plate, advance_processing,
    add_ingredient_to_plate, create_plate,
    get_ingredient_display_name
)
from ..core.recipes import OrderQueue
from .reward_shaper import RewardValues, RewardShaper
from .interactions.station_registry import StationRegistry
from .interactions.chopping_handler import ChoppingBoardHandler
from .interactions.dispenser_handler import PlateDispenserHandler, IngredientDispenserHandler
from .interactions.counter_handler import CounterHandler
from .interactions.trash_handler import TrashHandler
from .interactions.delivery_handler import DeliveryHandler


class InteractionHandler:
    """
    Coordinates and handles all interaction logic for agents.
    
    Delegates station-specific actions to registered StationHandler strategies.
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
        
        # Station Strategy Registry
        self.station_registry = StationRegistry()
        self._chopping_handler = ChoppingBoardHandler(self)
        self._plate_handler = PlateDispenserHandler(self)
        self._ingredient_handler = IngredientDispenserHandler(self)
        self._counter_handler = CounterHandler(self)
        self._trash_handler = TrashHandler(self)
        self._delivery_handler = DeliveryHandler(self)
        
        self.station_registry.register('C', self._chopping_handler)
        self.station_registry.register('P', self._plate_handler)
        self.station_registry.register('I', self._ingredient_handler)
        self.station_registry.register('#', self._counter_handler)
        self.station_registry.register('T', self._trash_handler)
        self.station_registry.register('D', self._delivery_handler)
        
    @property
    def rewards(self) -> RewardValues:
        """Centralized rewards properties for backward compatibility."""
        return self.reward_shaper.rewards

    def register_station_handler(self, station_char: str, handler):
        """Allows dynamic registration of custom station strategies."""
        self.station_registry.register(station_char, handler)

    def calculate_step_shaping(
        self,
        agent_id: str,
        game_state: Dict,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int]
    ) -> float:
        """Delegates step-level movement shaping (e.g. potential-based delivery guidance) to RewardShaper."""
        return self.reward_shaper.calculate_step_shaping(agent_id, game_state, old_pos, new_pos)

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
            
        # 1. Ingredient dispenser from map metadata - ONLY pickup, NEVER place
        if tile_pos in game_state.get("dispensers", {}):
            if carrying_item:
                if verbose:
                    print("  Cannot place items on ingredient dispenser")
                return 0.0, 0.0
            elif not ground_item:
                return self._handle_ingredient_dispenser(
                    agent_id, tile_pos, carrying_item,
                    game_state, verbose
                )
            else:
                if verbose:
                    print("  Invalid state: item on dispenser ground")
                return 0.0, 0.0

        # 2. Lookup station handler dynamically (supporting both ItemManager and mocks)
        if isinstance(station_type, str):
            station_char = station_type
        else:
            station_char = None
            if hasattr(self.item_manager, "NUM_TO_CHAR") and isinstance(self.item_manager.NUM_TO_CHAR, dict) and station_type in self.item_manager.NUM_TO_CHAR:
                station_char = self.item_manager.NUM_TO_CHAR[station_type]
            elif hasattr(self.item_manager, "get_cell_char"):
                res = self.item_manager.get_cell_char(station_type)
                if isinstance(res, str) and res != '?':
                    station_char = res
            
            if station_char is None and hasattr(self.item_manager, "get_cell_code"):
                code_to_char = {
                    self.item_manager.get_cell_code('C'): 'C',
                    self.item_manager.get_cell_code('P'): 'P',
                    self.item_manager.get_cell_code('D'): 'D',
                    self.item_manager.get_cell_code('T'): 'T',
                    self.item_manager.get_cell_code('#'): '#',
                }
                station_char = code_to_char.get(station_type)

        handler = self.station_registry.get(station_char) if station_char else None
        if handler is not None:
            return handler.handle(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )

        # 3. Invalid interaction location
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

    # =========================================================================
    # Backward compatibility adapter methods for tests & subclasses
    # =========================================================================

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
        return self._chopping_handler.handle(
            agent_id, tile_pos, carrying_item, carrying_instance_id,
            ground_item, ground_instance_id, game_state, verbose
        )

    def _handle_plate_dispenser(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        tile_pos = self._get_tile_in_front(agent_id, game_state)
        return self._plate_handler.handle(
            agent_id, tile_pos, carrying_item, carrying_instance_id,
            None, None, game_state, verbose
        )

    def _handle_ingredient_dispenser(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        return self._ingredient_handler.handle(
            agent_id, tile_pos, carrying_item, None,
            None, None, game_state, verbose
        )

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
        return self._counter_handler.handle(
            agent_id, tile_pos, carrying_item, carrying_instance_id,
            ground_item, ground_instance_id, game_state, verbose
        )

    def _handle_trash(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        tile_pos = self._get_tile_in_front(agent_id, game_state)
        return self._trash_handler.handle(
            agent_id, tile_pos, carrying_item, carrying_instance_id,
            None, None, game_state, verbose
        )

    def _handle_delivery(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool
    ) -> Tuple[float, float]:
        tile_pos = self._get_tile_in_front(agent_id, game_state)
        return self._delivery_handler.handle(
            agent_id, tile_pos, carrying_item, carrying_instance_id,
            None, None, game_state, verbose
        )

    def _is_correct_order(self, recipe_name: str, game_state: Dict) -> bool:
        """Check if delivered recipe is in correct order."""
        next_pending = OrderQueue.get_next_pending(
            game_state.get("recipe_order", []),
            game_state.get("delivered_recipes", [])
        )
        return next_pending is not None and recipe_name == next_pending

    # =========================================================================
    # Item management helper methods
    # =========================================================================

    def _get_tile_in_front(self, agent_id: str, game_state: Dict) -> Tuple[int, int]:
        """Get the tile position in front of the agent."""
        agent_pos = game_state["agent_pos"][agent_id]
        x, y = agent_pos
        
        dx, dy = game_state["agent_orientation"][agent_id]
        new_x, new_y = x + dx, y + dy
        
        map_height, map_width = game_state["map_static"].shape
        if 0 <= new_x < map_width and 0 <= new_y < map_height:
            return (new_x, new_y)
        
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
