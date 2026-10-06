import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from ..core.items import ItemType, ItemState, is_ready

@dataclass
class RewardValues:
    """
    Centralized reward values configuration for Cycle 3 (Milestone Reward Shaping).
    
    Milestone Progression Philosophy:
    - Step cost / Living cost (-0.01) to encourage efficiency and penalize idling/loops.
    - Pick up needed ingredient: +3.0 (unneeded: -2.0)
    - Place raw ingredient on chopping board: +5.0
    - Chop action: +1.0 per cut, +8.0 milestone when chopping is fully completed
    - Pick up empty plate when chopped ingredients exist: +5.0
    - Combine ready ingredient with plate: +15.0
    - Plate completed with all recipe ingredients: +20.0 milestone
    - Successful recipe delivery: Recipe reward (100.0 - 200.0) + order bonus (50.0)
    - Episode victory (all orders delivered): +100.0
    - Penalties for wasteful/cyclic actions (empty/wrong delivery, useless cycles, trash).
    """
    # Step living cost (encourages fastest path)
    step_cost: float = -0.005

    # Basic movements / placements
    pickup: float = 0.0
    place: float = 0.0
    place_on_chop: float = 5.0

    # Chopping / Processing
    chop_cut: float = 1.0
    chop_completed: float = 8.0
    process: float = 1.0  # legacy alias for chop_cut

    # Assembly / Combining
    combine: float = 0.0
    progress_pickup_needed: float = 0.0
    progress_place_on_chop: float = 5.0
    progress_pickup_ready: float = 2.0
    progress_combine_ready: float = 10.0
    progress_pickup_complete_plate: float = 5.0
    plate_complete_bonus: float = 15.0
    progress_pickup_plate_with_ready: float = 3.0
    progress_ready_to_combine: float = 3.0
    progress_partial_plate: float = 3.0
    progress_ingredient_added_base: float = 2.0
    progress_ingredient_added_increment: float = 2.0

    # Deliveries (Direct Sparse Primary Objective)
    delivery_base: float = 0.0
    order_bonus: float = 0.0
    order_bonus_mult: float = 0.0
    sequence_complete_bonus: float = 0.0
    episode_complete_bonus: float = 0.0

    # Multipliers
    delivery_multiplier_2nd: float = 1.0
    delivery_multiplier_3rd: float = 1.0
    delivery_multiplier_4th_plus: float = 1.0

    # Proximity & Attempts (disabled)
    delivery_proximity_reward: float = 0.0
    delivery_attempt_reward: float = 0.0
    delivery_distance_max: int = 5

    # Penalties (Anti-Reward Hacking & Efficiency)
    invalid: float = -0.01
    invalid_delivery_penalty: float = -5.0

    trash_penalty: float = -10.0
    trash_ready_penalty: float = -30.0
    trash_dish_penalty: float = -60.0
    unneeded_pickup_penalty: float = -2.0
    useless_cycle_penalty: float = -2.0
    useless_cycle: float = -2.0
    repetition_penalty: float = -1.0
    accumulation_penalty: float = 0.0
    accumulation_threshold: int = 5
    counter_clutter_penalty: float = -1.0


class RewardShaper:
    """
    Computes rewards and penalties to guide agents in the kitchen environment.
    """
    def __init__(self, item_manager, recipe_manager, rewards: Optional[RewardValues] = None):
        self.item_manager = item_manager
        self.recipe_manager = recipe_manager
        self.rewards = rewards or RewardValues()

    def _get_pending_recipes(self, game_state: Dict) -> List[str]:
        """Get list of remaining uncompleted recipe names in FIFO order."""
        recipe_order = game_state.get("recipe_order", [])
        if not recipe_order:
            return []
        pending = list(recipe_order)
        for delivered in game_state.get("delivered_recipes", []):
            if delivered in pending:
                pending.remove(delivered)
        return pending

    def _get_next_pending_recipe(self, game_state: Dict):
        """Get the next pending recipe and its required ingredients."""
        pending_orders = self._get_pending_recipes(game_state)
        if not pending_orders:
            return None, None, []
        
        next_recipe_name = pending_orders[0]
        recipe = self.recipe_manager.get_recipe_by_name(next_recipe_name)
        if recipe:
            return next_recipe_name, recipe, recipe.ingredients.copy()
        
        return None, None, []

    def _calculate_ready_to_combine_reward(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        ground_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """Deprecated proximity helper: returns 0.0 to prevent exploitation loops."""
        return 0.0

    def _calculate_delivery_proximity_reward(
        self,
        agent_id: str,
        carrying_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """Deprecated proximity helper: returns 0.0 to prevent exploitation loops."""
        return 0.0

    def _calculate_distance_to_delivery(self, agent_pos: Tuple[int, int], game_state: Dict) -> float:
        """Calculate minimum distance from agent to any delivery station."""
        delivery_stations = self._get_delivery_stations(game_state)
        if not delivery_stations:
            return float('inf')
        
        ax, ay = agent_pos
        min_distance = float('inf')
        for dx, dy in delivery_stations:
            distance = abs(ax - dx) + abs(ay - dy)
            min_distance = min(min_distance, distance)
        return min_distance

    def _get_delivery_stations(self, game_state: Dict) -> list:
        """Get all delivery station positions."""
        delivery_code = self.item_manager.get_cell_code('D')
        delivery_stations = []
        map_static = game_state["map_static"]
        H, W = map_static.shape
        
        for y in range(H):
            for x in range(W):
                if map_static[y, x] == delivery_code:
                    delivery_stations.append((x, y))
        return delivery_stations

    def _count_processed_ingredients_on_ground(self, game_state: Dict) -> int:
        """Count how many processed (ready) ingredients are on the ground."""
        count = 0
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for instance_id in ground_instances.values():
            item = item_instances.get(instance_id)
            if item and item.get("type") == ItemType.INGREDIENT and is_ready(item):
                count += 1
        return count

    def _check_action_repetition(self, agent_id: str, action_type: str, game_state: Dict) -> bool:
        """Check if agent is repeating the same action type many times consecutively."""
        history = game_state.get("action_history", {}).get(agent_id, [])
        if len(history) < 5:
            return False
        
        recent_actions = [a[0] for a in history[-5:]]
        same_action_count = sum(1 for a in recent_actions if a == action_type)
        return same_action_count >= 4

    def _check_ready_ingredient_nearby(self, agent_id: str, game_state: Dict) -> bool:
        """Check if there are ready ingredients anywhere on ground / chopping board."""
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for instance_id in ground_instances.values():
            item = item_instances.get(instance_id)
            if item and item.get("type") == ItemType.INGREDIENT and is_ready(item):
                return True
        return False

    def _calculate_partial_plate_reward(self, plate: Dict, game_state: Dict) -> float:
        """Calculate reward for partially complete plate."""
        if not plate or plate.get("type") != ItemType.DISH:
            return 0.0
        
        plate_ingredients = plate.get("ingredients", [])
        if not plate_ingredients:
            return 0.0
        
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        if not recipe or not required_ingredients:
            return 0.0
        
        required_set = set(ing.lower() for ing in required_ingredients)
        plate_set = set(ing.lower() for ing in plate_ingredients)
        
        matching_count = len(required_set.intersection(plate_set))
        total_required = len(required_ingredients)
        
        if 0 < matching_count < total_required:
            completion_ratio = matching_count / total_required
            return self.rewards.progress_partial_plate * completion_ratio
        
        return 0.0

    def _check_ingredient_already_processed(self, ingredient_name: str, game_state: Dict) -> bool:
        """Check if this ingredient has already been processed and is ready somewhere."""
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
        """Calculate milestone progress reward based on active recipe alignment."""
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        if not recipe or not required_ingredients:
            return 0.0
        
        progress_reward = 0.0
        req_lower = [ing.lower() for ing in required_ingredients]
        
        # Action: Pickup ingredient from dispenser
        if action_type == 'pickup_ingredient' and carrying_item:
            ingredient_name = carrying_item.get("name", "").lower()
            if ingredient_name in req_lower:
                if self._check_ingredient_already_processed(ingredient_name, game_state):
                    progress_reward += self.rewards.unneeded_pickup_penalty
                else:
                    progress_reward += self.rewards.progress_pickup_needed
            else:
                # Check if it belongs to ANY pending recipe
                all_pending_reqs = []
                for p_name in self._get_pending_recipes(game_state):
                    p_rec = self.recipe_manager.get_recipe_by_name(p_name)
                    if p_rec:
                        all_pending_reqs.extend([ing.lower() for ing in p_rec.ingredients])
                if ingredient_name not in all_pending_reqs:
                    progress_reward += self.rewards.unneeded_pickup_penalty
        
        # Action: Place raw ingredient on chopping board
        elif action_type in ('place_on_chop', 'place_raw_on_chop'):
            item = carrying_item or ground_item
            if item:
                ingredient_name = item.get("name", "").lower()
                if ingredient_name in req_lower:
                    progress_reward += self.rewards.progress_place_on_chop
        
        # Action: Pickup ready ingredient from chopping board / counter
        elif action_type in ('pickup_ready', 'pickup_ready_ingredient'):
            item = carrying_item or ground_item
            if item and is_ready(item):
                ingredient_name = item.get("name", "").lower()
                if ingredient_name in req_lower:
                    progress_reward += self.rewards.progress_pickup_ready
        
        # Action: Combine ingredient with plate
        elif action_type == 'combine':
            plate = carrying_item if carrying_item and carrying_item.get("type") == ItemType.DISH else ground_item
            ing = ground_item if carrying_item and carrying_item.get("type") == ItemType.DISH else carrying_item
            
            if ing and is_ready(ing):
                ing_name = ing.get("name", "").lower()
                if ing_name in req_lower:
                    progress_reward += self.rewards.progress_combine_ready
                    
            if plate and plate.get("type") == ItemType.DISH:
                plate_ingredients = plate.get("ingredients", [])
                matching_count = sum(1 for i in plate_ingredients if i.lower() in req_lower)
                progress_reward += (self.rewards.progress_ingredient_added_base +
                                    matching_count * self.rewards.progress_ingredient_added_increment)
                # Check if plate is fully assembled for current recipe
                if sorted([i.lower() for i in plate_ingredients]) == sorted(req_lower):
                    progress_reward += self.rewards.plate_complete_bonus
        
        # Action: Pickup complete plate
        elif action_type == 'pickup_plate_complete' and carrying_item:
            if carrying_item.get("type") == ItemType.DISH:
                plate_ingredients = carrying_item.get("ingredients", [])
                if sorted([i.lower() for i in plate_ingredients]) == sorted(req_lower):
                    progress_reward += self.rewards.progress_pickup_complete_plate
        
        # Apply MAPPO Reward Shaping Annealing (exponential decay as recipes are delivered)
        num_delivered = len(game_state.get("delivered_recipes", []))
        annealing_decay = math.exp(-num_delivered * 0.4)
        return progress_reward * annealing_decay
