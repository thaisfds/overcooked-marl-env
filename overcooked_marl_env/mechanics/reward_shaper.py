from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from ..core.items import ItemType, ItemState, is_ready

@dataclass
class RewardValues:
    """
    Centralized reward values configuration.
    
    Optimized philosophy: guide the agent to productive behaviors.
    - Basic actions (pickup/place): moderate positive reward
    - Transformation actions (process/combine): very high reward
    - Delivery actions (completing recipes): maximum reward
    - Repetition / accumulation penalties to prevent loops.
    """
    pickup: float = 1.0           
    place: float = 1.0            
    place_on_chop: float = 20.0   
    
    process: float = 50.0         
    combine: float = 100.0        
    
    delivery_base: float = 250.0  
    order_bonus_mult: float = 1.0 
    
    progress_pickup_needed: float = 3.0      
    progress_place_on_chop: float = 4.0      
    progress_pickup_ready: float = 3.5       
    progress_combine_ready: float = 12.0     
    progress_pickup_complete_plate: float = 4.0  
    
    progress_ingredient_added_base: float = 8.0  
    progress_ingredient_added_increment: float = 3.0  
    
    progress_pickup_plate_with_ready: float = 5.0  
    progress_ready_to_combine: float = 3.0        
    progress_partial_plate: float = 2.0            
    
    delivery_proximity_reward: float = 1.0   
    delivery_attempt_reward: float = 0.5     
    delivery_distance_max: int = 5           
    
    repetition_penalty: float = -5.0         
    accumulation_penalty: float = -3.0        
    accumulation_threshold: int = 3          
    
    sequence_complete_bonus: float = 30.0    
    
    delivery_multiplier_2nd: float = 1.1    
    delivery_multiplier_3rd: float = 1.2    
    delivery_multiplier_4th_plus: float = 1.3  
    
    invalid: float = 0.0          
    useless_cycle: float = 0.0    
    trash_penalty: float = 0.0   
    counter_clutter_penalty: float = -5.0  


class RewardShaper:
    """
    Computes rewards and penalties to guide agents in the kitchen environment.
    """
    def __init__(self, item_manager, recipe_manager, rewards: Optional[RewardValues] = None):
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
        """Calculate reward when agent is ready to combine."""
        if carrying_item and is_ready(carrying_item) and ground_item:
            if ground_item.get("type") == ItemType.DISH:
                return self.rewards.progress_ready_to_combine
        
        if carrying_item and carrying_item.get("type") == ItemType.DISH and ground_item:
            if is_ready(ground_item):
                return self.rewards.progress_ready_to_combine
        
        if carrying_item and is_ready(carrying_item):
            agent_pos = game_state["agent_pos"][agent_id]
            ax, ay = agent_pos
            
            ground_instances = game_state.get("ground_instances", {})
            item_instances = game_state.get("item_instances", {})
            
            for (x, y), instance_id in ground_instances.items():
                distance = abs(ax - x) + abs(ay - y)
                if distance <= 2:
                    item = item_instances.get(instance_id)
                    if item and item.get("type") == ItemType.DISH:
                        return self.rewards.progress_ready_to_combine
        
        if carrying_item and carrying_item.get("type") == ItemType.DISH:
            agent_pos = game_state["agent_pos"][agent_id]
            ax, ay = agent_pos
            
            ground_instances = game_state.get("ground_instances", {})
            item_instances = game_state.get("item_instances", {})
            
            for (x, y), instance_id in ground_instances.items():
                distance = abs(ax - x) + abs(ay - y)
                if distance <= 2:
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
        """Calculate proximity reward if agent is near delivery station with complete plate."""
        if not carrying_item or carrying_item.get("type") != ItemType.DISH:
            return 0.0
        
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            return 0.0
        
        is_valid, _, _ = self.recipe_manager.validate_delivery(ingredients)
        if not is_valid:
            return 0.0
        
        agent_pos = game_state["agent_pos"][agent_id]
        distance = self._calculate_distance_to_delivery(agent_pos, game_state)
        
        if distance <= self.rewards.delivery_distance_max:
            proximity_factor = 1.0 - (distance / self.rewards.delivery_distance_max)
            return self.rewards.delivery_proximity_reward * proximity_factor
        
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
        """Check if agent is repeating the same action type many times."""
        history = game_state.get("action_history", {}).get(agent_id, [])
        if len(history) < 5:
            return False
        
        recent_actions = [a[0] for a in history[-5:]]
        same_action_count = sum(1 for a in recent_actions if a == action_type)
        return same_action_count >= 4

    def _check_ready_ingredient_nearby(self, agent_id: str, game_state: Dict) -> bool:
        """Check if there are ready ingredients on ground near the agent."""
        agent_pos = game_state["agent_pos"][agent_id]
        ax, ay = agent_pos
        
        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})
        
        for (x, y), instance_id in ground_instances.items():
            distance = abs(ax - x) + abs(ay - y)
            if distance <= 2:
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

    def _get_next_pending_recipe(self, game_state: Dict):
        """Get the next pending recipe and its required ingredients."""
        recipe_order = game_state.get("recipe_order", [])
        if not recipe_order:
            return None, None, []
        
        pending_orders = list(recipe_order)
        for delivered_recipe in game_state.get("delivered_recipes", []):
            if delivered_recipe in pending_orders:
                pending_orders.remove(delivered_recipe)
        
        if not pending_orders:
            return None, None, []
        
        next_recipe_name = pending_orders[0]
        recipe = self.recipe_manager.get_recipe_by_name(next_recipe_name)
        if recipe:
            return next_recipe_name, recipe, recipe.ingredients.copy()
        
        return None, None, []

    def _calculate_progress_reward(
        self,
        action_type: str,
        carrying_item: Optional[Dict],
        ground_item: Optional[Dict],
        game_state: Dict
    ) -> float:
        """Calculate progress reward based on recipe alignment."""
        recipe_name, recipe, required_ingredients = self._get_next_pending_recipe(game_state)
        if not recipe or not required_ingredients:
            return 0.0
        
        progress_reward = 0.0
        
        # Action: Pickup ingredient from dispenser
        if action_type == 'pickup_ingredient' and carrying_item:
            ingredient_name = carrying_item.get("name", "").lower()
            if ingredient_name in [ing.lower() for ing in required_ingredients]:
                progress_reward += self.rewards.progress_pickup_needed
        
        # Action: Place raw ingredient on chopping board
        elif action_type == 'place_raw_on_chop' and ground_item:
            ingredient_name = ground_item.get("name", "").lower()
            if ingredient_name in [ing.lower() for ing in required_ingredients]:
                progress_reward += self.rewards.progress_place_on_chop
        
        # Action: Pickup ready ingredient from chopping board
        elif action_type == 'pickup_ready_ingredient' and carrying_item:
            if is_ready(carrying_item):
                ingredient_name = carrying_item.get("name", "").lower()
                if ingredient_name in [ing.lower() for ing in required_ingredients]:
                    progress_reward += self.rewards.progress_pickup_ready
        
        # Action: Combine ingredient with plate
        elif action_type == 'combine' and carrying_item:
            if is_ready(carrying_item):
                ingredient_name = carrying_item.get("name", "").lower()
                if ingredient_name in [ing.lower() for ing in required_ingredients]:
                    progress_reward += self.rewards.progress_combine_ready
                    if ground_item and ground_item.get("type") == ItemType.DISH:
                        plate_ingredients = ground_item.get("ingredients", [])
                        matching_count = sum(1 for ing in plate_ingredients 
                                           if ing.lower() in [r.lower() for r in required_ingredients])
                        scaled_reward = (self.rewards.progress_ingredient_added_base + 
                                       matching_count * self.rewards.progress_ingredient_added_increment)
                        progress_reward += scaled_reward
            elif carrying_item.get("type") == ItemType.DISH:
                if ground_item and is_ready(ground_item):
                    ingredient_name = ground_item.get("name", "").lower()
                    if ingredient_name in [ing.lower() for ing in required_ingredients]:
                        progress_reward += self.rewards.progress_combine_ready
                        plate_ingredients = carrying_item.get("ingredients", [])
                        matching_count = sum(1 for ing in plate_ingredients 
                                           if ing.lower() in [r.lower() for r in required_ingredients])
                        scaled_reward = (self.rewards.progress_ingredient_added_base + 
                                       matching_count * self.rewards.progress_ingredient_added_increment)
                        progress_reward += scaled_reward
        
        # Action: Pickup complete plate
        elif action_type == 'pickup_plate_complete' and carrying_item:
            if carrying_item.get("type") == ItemType.DISH:
                plate_ingredients = carrying_item.get("ingredients", [])
                if sorted([ing.lower() for ing in plate_ingredients]) == sorted([ing.lower() for ing in required_ingredients]):
                    progress_reward += self.rewards.progress_pickup_complete_plate
        
        return progress_reward
