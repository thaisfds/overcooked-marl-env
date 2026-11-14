import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from ..core.items import ItemType, ItemState, is_ready, is_plate
from ..core.recipes import OrderQueue

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
    place_on_chop: float = 3.0

    # Chopping / Processing
    chop_cut: float = 0.5
    chop_completed: float = 5.0
    process: float = 0.5  # alias for chop_cut

    # Assembly / Combining
    combine: float = 0.0
    progress_pickup_needed: float = 3.0
    progress_place_on_chop: float = 3.0
    progress_pickup_ready: float = 2.0
    progress_combine_ready: float = 8.0
    progress_pickup_complete_plate: float = 10.0
    plate_complete_bonus: float = 20.0
    progress_pickup_plate_with_ready: float = 3.0
    progress_ready_to_combine: float = 3.0
    progress_partial_plate: float = 3.0
    progress_ingredient_added_base: float = 2.0
    progress_ingredient_added_increment: float = 2.0

    # Deliveries (Direct Sparse Primary Objective)
    delivery_base: float = 0.0
    order_bonus: float = 0.0
    order_bonus_mult: float = 1.0          # Reinforced: 100% of recipe reward when delivered strictly in FIFO HUD top order
    out_of_order_penalty: float = 0.0      # Out of order receives 0 order bonus (no extra penalty, correct order gives 2x advantage)
    sequence_complete_bonus: float = 50.0
    episode_complete_bonus: float = 5000.0  # Huge 5000pt victory bonus when 0 pending dishes remain!

    # Dynamic Multipliers (Formula: 1.0 + step * (k - 1) for the k-th delivery)
    delivery_multiplier_step: float = 0.5   # Dynamic +50% multiplier per extra delivery (1st=1.0x, 2nd=1.5x, 3rd=2.0x, N-th=1+0.5*(N-1)x)
    delivery_multiplier_2nd: float = 1.0    # Backwards compatibility alias
    delivery_multiplier_3rd: float = 1.0    # Backwards compatibility alias
    delivery_multiplier_4th_plus: float = 1.0 # Backwards compatibility alias

    # Proximity & Delivery Guidance (Potential-based shaping when holding deliverable dish)
    delivery_guidance_weight: float = 1.0  # Reward per unit of distance closed towards delivery
    delivery_proximity_reward: float = 0.0
    delivery_attempt_reward: float = 0.0
    delivery_distance_max: int = 5

    # Multi-Agent Cooperative Handoff & Fair Credit Assignment
    cooperative_handoff: float = 5.0          # Reward when an agent picks up an item placed on counter by teammate
    handoff_place: float = 1.0                # Small positive nudge when placing needed item on counter
    cooperative_delivery_bonus: float = 50.0  # Extra team bonus when both agents contributed to a delivered recipe
    solo_delivery_bonus: float = 0.0          # Bonus for delivering agent on solo deliveries
    inactivity_penalty: float = 0.0           # 0.0 by default to prevent penalizing agents waiting for teammates
    inactivity_threshold: int = 40            # Steps before inactivity penalty kicks in (if enabled)

    # Penalties (Anti-Reward Hacking & Efficiency)
    invalid: float = -0.01
    invalid_delivery_penalty: float = -2.0
    unrequested_delivery_penalty: float = -10.0
    abandon_dish_penalty: float = -10.0

    trash_penalty: float = -10.0
    trash_ready_penalty: float = -30.0
    trash_dish_penalty: float = -60.0
    unneeded_pickup_penalty: float = -2.0
    useless_cycle_penalty: float = -1.0
    useless_cycle: float = -1.0
    repetition_penalty: float = 0.0
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
        return OrderQueue.get_pending(
            game_state.get("recipe_order", []),
            game_state.get("delivered_recipes", [])
        )

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

    def _find_counter_with_deliverable_dish(self, agent_pos: Tuple[int, int], game_state: Dict) -> Optional[Tuple[int, int]]:
        """Find position of nearest counter holding a plate with a valid pending recipe."""
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        pending_recipes = self._get_pending_recipes(game_state)
        if not pending_recipes:
            return None

        deliverable_counters = []
        for pos, iid in ground_instances.items():
            item = item_instances.get(iid)
            if item and is_plate(item):
                ings = item.get("ingredients", [])
                if ings:
                    is_valid, r_name, _ = self.recipe_manager.validate_delivery(ings)
                    if is_valid and r_name in pending_recipes:
                        deliverable_counters.append(pos)
        
        if not deliverable_counters:
            return None
        return min(deliverable_counters, key=lambda c: abs(c[0] - agent_pos[0]) + abs(c[1] - agent_pos[1]))

    def calculate_step_shaping(
        self,
        agent_id: str,
        game_state: Dict,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int]
    ) -> float:
        """
        Calculates exploit-free potential-based directional shaping for movements (Ng et al., 1999).
        
        1. If agent carries a plate with a valid pending recipe:
           Awards +weight * (dist_old - dist_new) towards nearest delivery station.
        2. If agent is empty-handed and a deliverable completed dish is on a counter:
           Awards +weight * (dist_old - dist_new) towards that counter.
        
        Net reward across round-trips is strictly zero, preventing exploitation loops.
        """
        weight = getattr(self.rewards, "delivery_guidance_weight", 0.0)
        if weight <= 0.0 or old_pos == new_pos:
            return 0.0

        # Check what agent is carrying
        holding_instance = game_state.get("agent_holding_instance", {}).get(agent_id)
        if holding_instance is not None:
            item = game_state.get("item_instances", {}).get(holding_instance)
            if item and is_plate(item):
                ingredients = item.get("ingredients", [])
                if ingredients:
                    # Check if plate matches any currently pending recipe in HUD queue
                    is_valid, recipe_name, _ = self.recipe_manager.validate_delivery(ingredients)
                    if is_valid:
                        pending_recipes = self._get_pending_recipes(game_state)
                        if recipe_name in pending_recipes:
                            dist_old = self._calculate_distance_to_delivery(old_pos, game_state)
                            dist_new = self._calculate_distance_to_delivery(new_pos, game_state)
                            if not (math.isinf(dist_old) or math.isinf(dist_new)):
                                return weight * (dist_old - dist_new)

        # If agent is empty-handed, guide towards counter with a completed deliverable dish if one exists
        elif holding_instance is None:
            target_counter = self._find_counter_with_deliverable_dish(old_pos, game_state)
            if target_counter is not None:
                dist_old = abs(old_pos[0] - target_counter[0]) + abs(old_pos[1] - target_counter[1])
                dist_new = abs(new_pos[0] - target_counter[0]) + abs(new_pos[1] - target_counter[1])
                return weight * (dist_old - dist_new)

        return 0.0

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
        """Calculate reward for partially complete plate across pending recipes."""
        if not plate or not is_plate(plate):
            return 0.0
        
        plate_ingredients = plate.get("ingredients", [])
        if not plate_ingredients:
            return 0.0
        
        plate_set = set(ing.lower() for ing in plate_ingredients)
        best_ratio = 0.0

        # Check next pending recipe (top priority)
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        if recipe and required_ingredients:
            required_set = set(ing.lower() for ing in required_ingredients)
            matching_count = len(required_set.intersection(plate_set))
            total_required = len(required_ingredients)
            if 0 < matching_count < total_required:
                best_ratio = matching_count / total_required

        # Fallback: check across all other pending recipes in queue
        if best_ratio == 0.0:
            for p_name in self._get_pending_recipes(game_state):
                p_rec = self.recipe_manager.get_recipe_by_name(p_name)
                if p_rec and p_rec.ingredients:
                    p_req_set = set(ing.lower() for ing in p_rec.ingredients)
                    m_count = len(p_req_set.intersection(plate_set))
                    t_req = len(p_rec.ingredients)
                    if 0 < m_count < t_req:
                        best_ratio = max(best_ratio, (m_count / t_req) * 0.75)

        if best_ratio > 0.0:
            return self.rewards.progress_partial_plate * best_ratio
        
        return 0.0

    def _check_ingredient_already_processed(
        self,
        ingredient_name: str,
        game_state: Dict,
        exclude_instance_id: Optional[int] = None,
        check_all_pending: bool = False
    ) -> bool:
        """
        Check if this ingredient is already available or in production across the kitchen.
        Counts items on ground, counters, boards, and inside existing plates vs total needed.
        """
        ingredient_name_lower = ingredient_name.lower()
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        holding_instances = game_state.get("agent_holding_instance", {})
        
        # Count existing instances in the environment
        existing_count = 0
        
        # 1. Check ground / counter / station instances
        for instance_id in ground_instances.values():
            if exclude_instance_id is not None and instance_id == exclude_instance_id:
                continue
            item = item_instances.get(instance_id)
            if not item:
                continue
            if item.get("type") == ItemType.INGREDIENT:
                if item.get("name", "").lower() == ingredient_name_lower:
                    existing_count += 1
            elif is_plate(item):
                p_ings = [i.lower() for i in item.get("ingredients", [])]
                if ingredient_name_lower in p_ings:
                    existing_count += 1

        # 2. Check agents holding items
        for instance_id in holding_instances.values():
            if instance_id is None:
                continue
            if exclude_instance_id is not None and instance_id == exclude_instance_id:
                continue
            item = item_instances.get(instance_id)
            if not item:
                continue
            if item.get("type") == ItemType.INGREDIENT:
                if item.get("name", "").lower() == ingredient_name_lower:
                    existing_count += 1
            elif is_plate(item):
                p_ings = [i.lower() for i in item.get("ingredients", [])]
                if ingredient_name_lower in p_ings:
                    existing_count += 1

        # How many are needed?
        needed_count = 0
        if check_all_pending:
            pending_recipes = self._get_pending_recipes(game_state)
            for p_name in pending_recipes:
                p_rec = self.recipe_manager.get_recipe_by_name(p_name)
                if p_rec and p_rec.ingredients:
                    needed_count += sum(1 for i in p_rec.ingredients if i.lower() == ingredient_name_lower)
        else:
            _, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
            if required_ingredients:
                needed_count = sum(1 for i in required_ingredients if i.lower() == ingredient_name_lower)

        # If we already have at least as many as needed, it's considered already processed/available
        return existing_count >= max(1, needed_count)

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
        pending_recipes = self._get_pending_recipes(game_state)
        all_pending_reqs = []
        for p_name in pending_recipes:
            p_rec = self.recipe_manager.get_recipe_by_name(p_name)
            if p_rec:
                all_pending_reqs.extend([ing.lower() for ing in p_rec.ingredients])
        
        # Action: Pickup ingredient from dispenser
        if action_type == 'pickup_ingredient' and carrying_item:
            ingredient_name = carrying_item.get("name", "").lower()
            exclude_id = None
            for i_id, i_item in game_state.get("item_instances", {}).items():
                if i_item is carrying_item:
                    exclude_id = i_id
                    break
            if ingredient_name in req_lower:
                if self._check_ingredient_already_processed(ingredient_name, game_state, exclude_instance_id=exclude_id, check_all_pending=False):
                    progress_reward += self.rewards.unneeded_pickup_penalty
                else:
                    progress_reward += self.rewards.progress_pickup_needed
            elif ingredient_name in all_pending_reqs:
                if self._check_ingredient_already_processed(ingredient_name, game_state, exclude_instance_id=exclude_id, check_all_pending=True):
                    progress_reward += self.rewards.unneeded_pickup_penalty
                else:
                    progress_reward += self.rewards.progress_pickup_needed * 0.75
            else:
                progress_reward += self.rewards.unneeded_pickup_penalty
        
        # Action: Place raw ingredient on chopping board or custom station
        elif action_type in ('place_on_chop', 'place_raw_on_chop', 'place_on_station'):
            item = carrying_item or ground_item
            if item:
                ingredient_name = item.get("name", "").lower()
                if ingredient_name in req_lower:
                    progress_reward += self.rewards.progress_place_on_chop
                elif ingredient_name in all_pending_reqs:
                    progress_reward += self.rewards.progress_place_on_chop * 0.75
        
        # Action: Pickup ready ingredient from chopping board / counter
        elif action_type in ('pickup_ready', 'pickup_ready_ingredient'):
            item = carrying_item or ground_item
            if item and is_ready(item):
                ingredient_name = item.get("name", "").lower()
                if ingredient_name in req_lower:
                    progress_reward += self.rewards.progress_pickup_ready
                elif ingredient_name in all_pending_reqs:
                    progress_reward += self.rewards.progress_pickup_ready * 0.75
        
        # Action: Combine ingredient with plate
        elif action_type == 'combine':
            plate = carrying_item if is_plate(carrying_item) else ground_item
            ing = ground_item if is_plate(carrying_item) else carrying_item
            
            if ing and is_ready(ing):
                ing_name = ing.get("name", "").lower()
                if ing_name in req_lower:
                    progress_reward += self.rewards.progress_combine_ready
                elif ing_name in all_pending_reqs:
                    progress_reward += self.rewards.progress_combine_ready * 0.75
                    
            if plate and is_plate(plate):
                plate_ingredients = plate.get("ingredients", [])
                matching_count_top = sum(1 for i in plate_ingredients if i.lower() in req_lower)
                matching_count_any = max(
                    (sum(1 for i in plate_ingredients if i.lower() in [r_ing.lower() for r_ing in p_rec.ingredients])
                     for p_name in pending_recipes
                     if (p_rec := self.recipe_manager.get_recipe_by_name(p_name))),
                    default=0
                )
                matching_count = max(matching_count_top, matching_count_any)
                progress_reward += (self.rewards.progress_ingredient_added_base +
                                    matching_count * self.rewards.progress_ingredient_added_increment)
                
                # Composite dish milestone bonus (2+ ingredients combined on plate)
                if len(plate_ingredients) >= 2:
                    progress_reward += 10.0
                
                # Check if plate is fully assembled for current recipe
                if sorted([i.lower() for i in plate_ingredients]) == sorted(req_lower):
                    progress_reward += self.rewards.plate_complete_bonus
                else:
                    # Check if plate completes ANY pending recipe in HUD queue
                    is_val, r_name, _ = self.recipe_manager.validate_delivery(plate_ingredients)
                    if is_val and r_name in pending_recipes:
                        progress_reward += self.rewards.plate_complete_bonus * 0.75
        
        # Action: Pickup complete plate
        elif action_type == 'pickup_plate_complete' and carrying_item:
            if is_plate(carrying_item):
                plate_ingredients = [i.lower() for i in carrying_item.get("ingredients", [])]
                if sorted(plate_ingredients) == sorted(req_lower):
                    progress_reward += self.rewards.progress_pickup_complete_plate
                else:
                    is_val, r_name, _ = self.recipe_manager.validate_delivery(plate_ingredients)
                    if is_val and r_name in pending_recipes:
                        progress_reward += self.rewards.progress_pickup_complete_plate * 0.75
        
        # Apply MAPPO Reward Shaping Annealing (gentle decay as recipes are delivered)
        num_delivered = len(game_state.get("delivered_recipes", []))
        annealing_decay = math.exp(-num_delivered * 0.15)
        return progress_reward * annealing_decay
