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
    
    Filosofia OTIMIZADA: Guiar o agente para comportamentos produtivos.
    - Ações básicas (pegar/colocar): recompensa positiva moderada
    - Ações de transformação (processar/combinar): recompensa MUITO ALTA
    - Ações de entrega (completar receitas): recompensa MÁXIMA
    - Recompensas progressivas: guiar sequência correta (pegar → picar → prato → entregar)
    - Penalidades estratégicas para comportamentos destrutivos
    """
    # Ações básicas (progresso inicial)
    pickup: float = 1.0           # Recompensa básica por pegar item
    place: float = 1.0            # Recompensa básica por colocar item
    place_on_chop: float = 20.0   # Recompensa especial para colocar na bancada
    
    # Ações de transformação (progresso REAL) - BALANCEADAS
    process: float = 50.0         # AUMENTADO de 30.0 → 50.0 (cortar é fundamental!)
    combine: float = 100.0        # AUMENTADO de 75.0 → 100.0 (combinar é MUITO importante!)
    
    # Recompensas de entrega (objetivo final) - BALANCEADAS
    delivery_base: float = 250.0  # AJUSTADO de 300.0 → 250.0 (era 200.0 originalmente)
    order_bonus_mult: float = 1.0 # Multiplicador de bônus por ordem correta
    
    # Recompensas progressivas (reward shaping) - AJUSTADAS para valores mais conservadores
    progress_pickup_needed: float = 3.0      # AJUSTADO de 5.0 → 3.0 (era 2.0 originalmente)
    progress_place_on_chop: float = 4.0      # AJUSTADO de 8.0 → 4.0 (era 3.0 originalmente)
    progress_pickup_ready: float = 3.5       # AJUSTADO de 6.0 → 3.5 (era 2.5 originalmente)
    progress_combine_ready: float = 12.0     # AJUSTADO de 25.0 → 12.0 (era 10.0 originalmente)
    progress_pickup_complete_plate: float = 4.0  # AJUSTADO de 10.0 → 4.0 (era 2.0 originalmente)
    
    # Recompensas escalonadas por ingrediente adicionado ao prato da receita correta
    progress_ingredient_added_base: float = 8.0  # AJUSTADO de 15.0 → 8.0 (base para primeiro ingrediente)
    progress_ingredient_added_increment: float = 3.0  # AJUSTADO de 5.0 → 3.0 (incremento por ingrediente adicional)
    
    # Recompensas específicas para Problema 3 (agente não coloca no prato)
    progress_pickup_plate_with_ready: float = 5.0  # Recompensa alta por pegar prato quando já tem ingrediente pronto
    progress_ready_to_combine: float = 3.0        # Recompensa quando está pronto para combinar (ingrediente pronto + próximo de prato)
    progress_partial_plate: float = 2.0            # Recompensa progressiva por prato parcialmente completo
    
    # Recompensas de entrega (Problema 1: agente não sabe entregar)
    delivery_proximity_reward: float = 1.0   # Recompensa por estar próximo da estação de entrega com prato completo
    delivery_attempt_reward: float = 0.5     # Recompensa por tentar entregar (mesmo que falhe)
    delivery_distance_max: int = 5           # Distância máxima para recompensa de proximidade
    
    # Penalidades por loops (Problema 2: agente fica preso em pegar/picar)
    repetition_penalty: float = -5.0         # Penalidade por repetir mesma ação muitas vezes
    accumulation_penalty: float = -3.0        # Penalidade por acumular muitos ingredientes processados sem usar
    accumulation_threshold: int = 3          # Número de ingredientes processados no chão antes de penalizar
    
    # Recompensas de sequência completa
    sequence_complete_bonus: float = 30.0    # AJUSTADO de 50.0 → 30.0 (era 25.0 originalmente)
    
    # Bônus por múltiplas entregas consecutivas - AJUSTADOS para valores mais conservadores
    delivery_multiplier_2nd: float = 1.1    # AJUSTADO de 1.2 → 1.1 (multiplicador para segunda entrega)
    delivery_multiplier_3rd: float = 1.2    # AJUSTADO de 1.5 → 1.2 (multiplicador para terceira entrega)
    delivery_multiplier_4th_plus: float = 1.3  # AJUSTADO de 2.0 → 1.3 (multiplicador para quarta+ entrega)
    
    # Penalidades estratégicas
    invalid: float = 0.0          # ZERADO - sem punição para exploração
    useless_cycle: float = 0.0    # ZERADO - sem punição para exploração
    trash_penalty: float = 0.0   # Penalidade por jogar item no lixo
    counter_clutter_penalty: float = -5.0  # Penalidade por acumular >3 itens em bancadas


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
    
    def _calculate_ready_to_combine_reward(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        ground_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """
        Calculate reward when agent is ready to combine (has ready ingredient and is near plate, or vice versa).
        
        Args:
            agent_id: ID of the agent
            carrying_item: Item the agent is carrying
            ground_item: Item on the ground
            game_state: Current game state dictionary
            
        Returns:
            Reward for being ready to combine (0.0 if conditions not met)
        """
        # Case 1: Carrying ready ingredient, ground has plate
        if carrying_item and is_ready(carrying_item) and ground_item:
            if ground_item.get("type") == ItemType.DISH:
                return self.rewards.progress_ready_to_combine
        
        # Case 2: Carrying plate, ground has ready ingredient
        if carrying_item and carrying_item.get("type") == ItemType.DISH and ground_item:
            if is_ready(ground_item):
                return self.rewards.progress_ready_to_combine
        
        # Case 3: Carrying ready ingredient, check if there's a plate nearby (within 2 tiles)
        if carrying_item and is_ready(carrying_item):
            agent_pos = game_state["agent_pos"][agent_id]
            ax, ay = agent_pos
            
            ground_instances = game_state.get("ground_instances", {})
            item_instances = game_state.get("item_instances", {})
            
            for (x, y), instance_id in ground_instances.items():
                distance = abs(ax - x) + abs(ay - y)
                if distance <= 2:  # Within 2 tiles
                    item = item_instances.get(instance_id)
                    if item and item.get("type") == ItemType.DISH:
                        return self.rewards.progress_ready_to_combine
        
        # Case 4: Carrying plate, check if there's a ready ingredient nearby
        if carrying_item and carrying_item.get("type") == ItemType.DISH:
            agent_pos = game_state["agent_pos"][agent_id]
            ax, ay = agent_pos
            
            ground_instances = game_state.get("ground_instances", {})
            item_instances = game_state.get("item_instances", {})
            
            for (x, y), instance_id in ground_instances.items():
                distance = abs(ax - x) + abs(ay - y)
                if distance <= 2:  # Within 2 tiles
                    item = item_instances.get(instance_id)
                    if item and item.get("type") == ItemType.INGREDIENT and is_ready(item):
                        return self.rewards.progress_ready_to_combine
        
        return 0.0
    
    def _calculate_delivery_proximity_reward(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """
        Calculate proximity reward if agent is near delivery station with complete plate.
        
        Args:
            agent_id: ID of the agent
            carrying_item: Item the agent is carrying
            game_state: Current game state dictionary
            
        Returns:
            Proximity reward (0.0 if conditions not met)
        """
        # Must be carrying a plate with ingredients
        if not carrying_item or carrying_item.get("type") != ItemType.DISH:
            return 0.0
        
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            return 0.0
        
        # Check if plate has valid recipe (complete plate)
        is_valid, _, _ = self.recipe_manager.validate_delivery(ingredients)
        if not is_valid:
            return 0.0
        
        # Calculate distance to delivery station
        agent_pos = game_state["agent_pos"][agent_id]
        distance = self._calculate_distance_to_delivery(agent_pos, game_state)
        
        # Give reward if within threshold distance
        if distance <= self.rewards.delivery_distance_max:
            # Reward decreases with distance
            proximity_factor = 1.0 - (distance / self.rewards.delivery_distance_max)
            return self.rewards.delivery_proximity_reward * proximity_factor
        
        return 0.0
    
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
        
        # Calculate proximity reward if agent is near delivery with complete plate
        proximity_reward = self._calculate_delivery_proximity_reward(agent_id, carrying_item, game_state)
        
        # Calculate "ready to combine" reward if agent has ready ingredient and is near plate (or vice versa)
        ready_to_combine_reward = self._calculate_ready_to_combine_reward(
            agent_id, carrying_item, ground_item, game_state
        )
        
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
            local_reward, global_reward = self._handle_chopping_board(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )
            return local_reward + proximity_reward + ready_to_combine_reward, global_reward
        
        # Plate dispenser
        elif station_type == plate_code:
            local_reward, global_reward = self._handle_plate_dispenser(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
            return local_reward + proximity_reward + ready_to_combine_reward, global_reward
        
        # Ingredient dispenser (from map) - ONLY pickup, NEVER place
        # Cannot place items on dispensers - only pick up ingredients when empty
        elif tile_pos in game_state["dispensers"]:
            if carrying_item:
                # Tentativa de colocar item no dispenser - bloqueado
                if verbose:
                    print(f"  Cannot place items on ingredient dispenser")
                return proximity_reward + ready_to_combine_reward, 0.0
            elif not ground_item:
                # Pickup de ingrediente do dispenser
                local_reward, global_reward = self._handle_ingredient_dispenser(
                    agent_id, tile_pos, carrying_item,
                    game_state, verbose
                )
                return local_reward + proximity_reward + ready_to_combine_reward, global_reward
            else:
                # Há item no chão do dispenser (não deveria acontecer)
                if verbose:
                    print(f"  Invalid state: item on dispenser ground")
                return proximity_reward + ready_to_combine_reward, 0.0
        
        # Delivery station
        elif station_type == delivery_code:
            local_reward, global_reward = self._handle_delivery(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
            return local_reward + proximity_reward + ready_to_combine_reward, global_reward
        
        # Trash station - delete items
        elif station_type == trash_code:
            local_reward, global_reward = self._handle_trash(
                agent_id, carrying_item, carrying_instance_id,
                game_state, verbose
            )
            return local_reward + proximity_reward + ready_to_combine_reward, global_reward
        
        # Counter/table (for placing/picking/combining items)
        # ONLY '#' counters - ingredient dispensers ('I') are handled above
        elif station_type == counter_code:
            local_reward, global_reward = self._handle_counter(
                agent_id, tile_pos, carrying_item, carrying_instance_id,
                ground_item, ground_instance_id, game_state, verbose
            )
            return local_reward + proximity_reward + ready_to_combine_reward, global_reward
        
        # Invalid interaction
        if verbose:
            print(f"  Invalid interaction location")
        return proximity_reward + ready_to_combine_reward, 0.0  # Retorna recompensas mesmo em interação inválida
    
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
            progress_reward = self._calculate_progress_reward(
                'place_on_chop', carrying_item, None, game_state
            )
            total_reward = self.rewards.place_on_chop + progress_reward
            
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
            self._update_item(ground_instance_id, updated_item, game_state)
            
            # Check for repetition penalty
            repetition_penalty = 0.0
            if self._check_action_repetition(agent_id, 'process', game_state):
                repetition_penalty = self.rewards.repetition_penalty
                if verbose:
                    print(f"  WARNING: Repetitive processing detected! Penalty: {repetition_penalty}")
            
            # Check for accumulation penalty (many processed ingredients on ground)
            accumulation_penalty = 0.0
            processed_count = self._count_processed_ingredients_on_ground(game_state)
            if processed_count >= self.rewards.accumulation_threshold:
                excess = processed_count - self.rewards.accumulation_threshold + 1
                accumulation_penalty = self.rewards.accumulation_penalty * excess
                if verbose:
                    print(f"  WARNING: Too many processed ingredients on ground ({processed_count})! Penalty: {accumulation_penalty}")
            
            total_reward = self.rewards.process + repetition_penalty + accumulation_penalty
            
            if verbose:
                cuts_done = updated_item.get("cuts_done", 0)
                cuts_needed = updated_item.get("cuts_needed", DEFAULT_CUTS_NEEDED)
                print(f"  Cutting... {cuts_done}/{cuts_needed} cuts")
                
                if is_ready(updated_item):
                    print(f"  Item processed and ready to use!")
            
            return total_reward, 0.0
        
        # Scenario 3: Pick up processed ingredient from chopping board
        elif not carrying_item and ground_item and is_ready(ground_item):
            # Calculate progress reward BEFORE moving item (use ground_item)
            progress_reward = self._calculate_progress_reward(
                'pickup_ready', ground_item, None, game_state
            )
            
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]
            
            total_reward = self.rewards.pickup + progress_reward
            
            if verbose:
                print(f"  Picked up {ground_item['name']} from chopping board (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Invalid chopping board interaction
        if verbose:
            print(f"  Invalid chopping board interaction")
        return 0.0, 0.0  # SEM PENALIDADE
    
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
            ready_ingredient_nearby = self._check_ready_ingredient_nearby(agent_id, game_state)
            progress_reward = 0.0
            if ready_ingredient_nearby:
                progress_reward = self.rewards.progress_ready_to_combine
                if verbose:
                    print(f"  Ready ingredient nearby - ready to combine!")
            
            total_reward = self.rewards.pickup + progress_reward
            
            if verbose:
                print(f"  Picked up an empty plate from dispenser (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} ready to combine)" if progress_reward > 0 else ""))
            return total_reward, 0.0
        
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
            
            # Calculate progress reward for combining
            progress_reward = self._calculate_progress_reward(
                'combine', carrying_item, plate, game_state
            )
            
            # Recompensa alta por pegar prato quando já tem ingrediente pronto
            pickup_plate_with_ready_reward = self.rewards.progress_pickup_plate_with_ready
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self._calculate_partial_plate_reward(plate, game_state)
            
            total_reward = (self.rewards.pickup + self.rewards.combine + 
                          progress_reward + pickup_plate_with_ready_reward + 
                          partial_plate_reward)
            
            if verbose:
                print(f"  Picked up plate and combined with {carrying_item['name']} (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      f" +{pickup_plate_with_ready_reward:.1f} pickup_plate_with_ready" +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Invalid plate dispenser interaction
        if verbose:
            print(f"  Cannot get plate: carrying non-ready ingredient")
        return 0.0, 0.0  # SEM PENALIDADE
    
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
            return 0.0, 0.0  # SEM PENALIDADE
        
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
        progress_reward = self._calculate_progress_reward(
            'pickup_ingredient', ingredient, None, game_state
        )
        total_reward = self.rewards.pickup + progress_reward
        
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
            updated_plate = add_ingredient_to_plate(ground_item, carrying_item["name"])
            
            # Delete ingredient
            self._delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            
            # Update plate on counter
            self._update_item(ground_instance_id, updated_plate, game_state)
            
            # Calculate progress reward
            progress_reward = self._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self._calculate_partial_plate_reward(updated_plate, game_state)
            
            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            
            if verbose:
                print(f"  Combined {carrying_item['name']} with plate on counter (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0
        
        # Scenario 2: Carrying plate + READY item on counter → pick up and combine
        elif is_carrying_dish and is_ground_ready:
            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"])
            self._update_item(carrying_instance_id, updated_plate, game_state)
            
            # Delete ingredient from counter
            self._delete_item_instance(ground_instance_id, game_state)
            del game_state["ground_instances"][(ix, iy)]
            
            # Calculate progress reward
            progress_reward = self._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            
            # Recompensa progressiva por prato parcialmente completo
            partial_plate_reward = self._calculate_partial_plate_reward(updated_plate, game_state)
            
            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            
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
                progress_reward = self._calculate_progress_reward(
                    'pickup_ready', ground_item, None, game_state
                )
            elif ground_item.get("type") == ItemType.DISH:
                # Picking up plate - check if it's complete
                progress_reward = self._calculate_progress_reward(
                    'pickup_plate_complete', ground_item, None, game_state
                )
            
            total_reward = self.rewards.pickup + progress_reward
            
            if verbose:
                print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            
            return total_reward, 0.0
        
        # Invalid counter interaction
        if verbose:
            print(f"  Invalid counter interaction")
        return 0.0, 0.0  # SEM PENALIDADE
    
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
                print(f"  Discarded {get_ingredient_display_name(carrying_item)} in trash (PENALIDADE: {self.rewards.trash_penalty})")
            return self.rewards.trash_penalty, 0.0  # PENALIDADE por desperdiçar item
        
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
            return 0.0, 0.0  # SEM PENALIDADE
        
        # Plate must have ingredients
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            if verbose:
                print(f"  Cannot deliver: plate is empty")
            return 0.0, 0.0  # SEM PENALIDADE
        
        # Validate recipe
        is_valid, recipe_name, recipe_reward = self.recipe_manager.validate_delivery(ingredients)
        
        if not is_valid:
            # Recompensa de tentativa mesmo que falhe (incentiva exploração)
            attempt_reward = self.rewards.delivery_attempt_reward
            if verbose:
                ingredients_str = ", ".join(ingredients)
                print(f"  Invalid recipe! Ingredients: [{ingredients_str}] (attempt reward: {attempt_reward})")
            return attempt_reward, 0.0
        
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
        
        # Bônus por completar sequência completa (pegar → picar → prato → entregar)
        sequence_bonus = self.rewards.sequence_complete_bonus
        
        # Bônus por múltiplas entregas consecutivas
        # Contar quantas receitas já foram entregues (incluindo esta)
        num_deliveries = len(game_state.get("delivered_recipes", []))
        if num_deliveries == 1:
            # Primeira entrega: sem multiplicador adicional
            delivery_multiplier = 1.0
        elif num_deliveries == 2:
            # Segunda entrega: multiplicador 1.2
            delivery_multiplier = self.rewards.delivery_multiplier_2nd
        elif num_deliveries == 3:
            # Terceira entrega: multiplicador 1.5
            delivery_multiplier = self.rewards.delivery_multiplier_3rd
        else:
            # Quarta+ entrega: multiplicador 2.0
            delivery_multiplier = self.rewards.delivery_multiplier_4th_plus
        
        # Aplicar multiplicador ao base_reward (não inclui order_bonus e sequence_bonus)
        delivery_bonus = base_reward * (delivery_multiplier - 1.0)  # Bônus adicional
        
        total_reward = base_reward + order_bonus + sequence_bonus + delivery_bonus
        global_reward_bonus = recipe_reward + order_bonus + delivery_bonus
        
        if verbose:
            if is_correct_order:
                print(f"  Recipe {recipe_name} delivered in correct order! (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Order Bonus: {order_bonus} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f})")
            else:
                print(f"  Recipe {recipe_name} delivered out of order (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f}) (no order bonus)")
        
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
    
    def _get_next_pending_recipe(self, game_state: Dict):
        """
        Get the next pending recipe and its required ingredients.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Tuple of (recipe_name, recipe_object, required_ingredients) or (None, None, [])
        """
        recipe_order = game_state.get("recipe_order", [])
        if not recipe_order:
            return None, None, []
        
        # Calculate pending orders
        pending_orders = list(recipe_order)
        for delivered_recipe in game_state.get("delivered_recipes", []):
            if delivered_recipe in pending_orders:
                pending_orders.remove(delivered_recipe)
        
        if not pending_orders:
            return None, None, []
        
        # Get first pending recipe
        next_recipe_name = pending_orders[0]
        recipe = self.recipe_manager.get_recipe_by_name(next_recipe_name)
        
        if recipe:
            return next_recipe_name, recipe, recipe.ingredients.copy()
        
        return None, None, []
    
    def _get_delivery_stations(self, game_state: Dict) -> list:
        """
        Get all delivery station positions.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            List of (x, y) tuples for delivery stations
        """
        delivery_code = self.item_manager.get_cell_code('D')
        delivery_stations = []
        
        map_static = game_state["map_static"]
        H, W = map_static.shape
        
        for y in range(H):
            for x in range(W):
                if map_static[y, x] == delivery_code:
                    delivery_stations.append((x, y))
        
        return delivery_stations
    
    def _calculate_distance_to_delivery(self, agent_pos: Tuple[int, int], game_state: Dict) -> float:
        """
        Calculate minimum distance from agent to any delivery station.
        
        Args:
            agent_pos: Agent position (x, y)
            game_state: Current game state dictionary
            
        Returns:
            Minimum Manhattan distance to nearest delivery station, or float('inf') if none found
        """
        delivery_stations = self._get_delivery_stations(game_state)
        if not delivery_stations:
            return float('inf')
        
        ax, ay = agent_pos
        min_distance = float('inf')
        
        for dx, dy in delivery_stations:
            distance = abs(ax - dx) + abs(ay - dy)  # Manhattan distance
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _count_processed_ingredients_on_ground(self, game_state: Dict) -> int:
        """
        Count how many processed (ready) ingredients are on the ground.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Number of ready ingredients on ground
        """
        count = 0
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for instance_id in ground_instances.values():
            item = item_instances.get(instance_id)
            if item and item.get("type") == ItemType.INGREDIENT and is_ready(item):
                count += 1
        
        return count
    
    def _check_action_repetition(self, agent_id: str, action_type: str, game_state: Dict) -> bool:
        """
        Check if agent is repeating the same action type many times.
        
        Args:
            agent_id: ID of the agent
            action_type: Type of action ('pickup', 'place', 'process', etc.)
            game_state: Current game state dictionary
            
        Returns:
            True if action is being repeated too many times
        """
        history = game_state.get("action_history", {}).get(agent_id, [])
        if len(history) < 5:
            return False
        
        # Check last 5 actions
        recent_actions = [a[0] for a in history[-5:]]
        # Count how many times the same action appears
        same_action_count = sum(1 for a in recent_actions if a == action_type)
        
        # If same action appears 4+ times in last 5, it's repetitive
        return same_action_count >= 4
    
    def _check_ready_ingredient_nearby(self, agent_id: str, game_state: Dict) -> bool:
        """
        Check if there are ready ingredients on ground near the agent.
        
        Args:
            agent_id: ID of the agent
            game_state: Current game state dictionary
            
        Returns:
            True if there's a ready ingredient nearby (within 2 tiles)
        """
        agent_pos = game_state["agent_pos"][agent_id]
        ax, ay = agent_pos
        
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        # Check tiles within 2 Manhattan distance
        for (x, y), instance_id in ground_instances.items():
            distance = abs(ax - x) + abs(ay - y)
            if distance <= 2:  # Within 2 tiles
                item = item_instances.get(instance_id)
                if item and item.get("type") == ItemType.INGREDIENT and is_ready(item):
                    return True
        
        return False
    
    def _calculate_partial_plate_reward(self, plate: Dict, game_state: Dict) -> float:
        """
        Calculate reward for partially complete plate.
        
        Args:
            plate: Plate dictionary
            game_state: Current game state dictionary
            
        Returns:
            Reward for partial plate completion
        """
        if not plate or plate.get("type") != ItemType.DISH:
            return 0.0
        
        plate_ingredients = plate.get("ingredients", [])
        if not plate_ingredients:
            return 0.0
        
        # Get next pending recipe
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        if not recipe or not required_ingredients:
            return 0.0
        
        # Check how many required ingredients are in the plate
        required_set = set(ing.lower() for ing in required_ingredients)
        plate_set = set(ing.lower() for ing in plate_ingredients)
        
        # Count matching ingredients
        matching_count = len(required_set.intersection(plate_set))
        total_required = len(required_ingredients)
        
        # If plate has some but not all ingredients, give partial reward
        if 0 < matching_count < total_required:
            # Reward proportional to completion
            completion_ratio = matching_count / total_required
            return self.rewards.progress_partial_plate * completion_ratio
        
        return 0.0
    
    def _check_ingredient_already_processed(self, ingredient_name: str, game_state: Dict) -> bool:
        """
        Check if this ingredient has already been processed and is ready somewhere.
        
        Args:
            ingredient_name: Name of the ingredient
            game_state: Current game state dictionary
            
        Returns:
            True if there's already a ready version of this ingredient
        """
        ingredient_name_lower = ingredient_name.lower()
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for instance_id in ground_instances.values():
            item = item_instances.get(instance_id)
            if item and item.get("type") == ItemType.INGREDIENT:
                if item.get("name", "").lower() == ingredient_name_lower and is_ready(item):
                    return True
        
        return False
    
    def _calculate_progress_reward(
        self,
        action_type: str,
        carrying_item: Optional[Dict],
        ground_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """
        Calculate progress reward based on whether the action aligns with the recipe goal.
        
        Args:
            action_type: Type of action ('pickup', 'place', 'process', 'combine')
            carrying_item: Item the agent is carrying (if any)
            ground_item: Item on the ground (if any)
            game_state: Current game state dictionary
            
        Returns:
            Additional progress reward (0.0 if no progress alignment)
        """
        # Get next pending recipe
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        
        if not recipe or not required_ingredients:
            return 0.0
        
        progress_reward = 0.0
        
        # Action: Pickup ingredient from dispenser
        if action_type == 'pickup_ingredient' and carrying_item:
            ingredient_name = carrying_item.get("name", "").lower()
            # Check if this ingredient is needed for the recipe
            if ingredient_name in [ing.lower() for ing in required_ingredients]:
                progress_reward += self.rewards.progress_pickup_needed
        
        # Action: Place ingredient on chopping board
        elif action_type == 'place_on_chop' and carrying_item:
            ingredient_name = carrying_item.get("name", "").lower()
            # Check if this ingredient is needed and can be processed
            if ingredient_name in [ing.lower() for ing in required_ingredients]:
                if can_process(carrying_item):
                    # Só dar recompensa progressiva se o ingrediente ainda não foi processado antes
                    if not self._check_ingredient_already_processed(ingredient_name, game_state):
                        progress_reward += self.rewards.progress_place_on_chop
        
        # Action: Pickup processed ingredient from chopping board
        elif action_type == 'pickup_ready' and carrying_item:
            if is_ready(carrying_item):
                ingredient_name = carrying_item.get("name", "").lower()
                # Check if this processed ingredient is needed
                if ingredient_name in [ing.lower() for ing in required_ingredients]:
                    # Verificar se o ingrediente já está no prato (se sim, não precisa pegar de novo)
                    # Por enquanto, sempre dar recompensa se é necessário
                    progress_reward += self.rewards.progress_pickup_ready
        
        # Action: Combine ingredient with plate
        elif action_type == 'combine' and carrying_item:
            # If carrying ready ingredient and combining with plate
            if is_ready(carrying_item):
                ingredient_name = carrying_item.get("name", "").lower()
                if ingredient_name in [ing.lower() for ing in required_ingredients]:
                    progress_reward += self.rewards.progress_combine_ready
                    # Recompensa escalonada por ingrediente adicionado ao prato da receita correta
                    if ground_item and ground_item.get("type") == ItemType.DISH:
                        plate_ingredients = ground_item.get("ingredients", [])
                        # Contar quantos ingredientes da receita já estão no prato (antes de adicionar)
                        matching_count = sum(1 for ing in plate_ingredients 
                                           if ing.lower() in [r.lower() for r in required_ingredients])
                        # Recompensa escalonada: primeiro=15.0, segundo=20.0, terceiro=25.0
                        scaled_reward = (self.rewards.progress_ingredient_added_base + 
                                       matching_count * self.rewards.progress_ingredient_added_increment)
                        progress_reward += scaled_reward
            # If carrying plate and combining with ready ingredient
            elif carrying_item.get("type") == ItemType.DISH:
                if ground_item and is_ready(ground_item):
                    ingredient_name = ground_item.get("name", "").lower()
                    if ingredient_name in [ing.lower() for ing in required_ingredients]:
                        progress_reward += self.rewards.progress_combine_ready
                        # Recompensa escalonada por ingrediente adicionado ao prato da receita correta
                        plate_ingredients = carrying_item.get("ingredients", [])
                        # Contar quantos ingredientes da receita já estão no prato (antes de adicionar)
                        matching_count = sum(1 for ing in plate_ingredients 
                                           if ing.lower() in [r.lower() for r in required_ingredients])
                        # Recompensa escalonada: primeiro=15.0, segundo=20.0, terceiro=25.0
                        scaled_reward = (self.rewards.progress_ingredient_added_base + 
                                       matching_count * self.rewards.progress_ingredient_added_increment)
                        progress_reward += scaled_reward
        
        # Action: Pickup complete plate (ready for delivery)
        elif action_type == 'pickup_plate_complete' and carrying_item:
            if carrying_item.get("type") == ItemType.DISH:
                plate_ingredients = carrying_item.get("ingredients", [])
                # Check if plate has all required ingredients for the recipe
                if sorted([ing.lower() for ing in plate_ingredients]) == sorted([ing.lower() for ing in required_ingredients]):
                    progress_reward += self.rewards.progress_pickup_complete_plate
        
        return progress_reward
    
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
