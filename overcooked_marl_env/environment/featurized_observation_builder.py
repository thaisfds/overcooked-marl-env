import numpy as np
from typing import Dict, Tuple, Any, List

class FeaturizedObservationBuilder:
    """
    Builds a dense, normalized, scale-invariant 1D vector representation of the game state.
    Vector shape is ALWAYS exactly (96,) regardless of map size, number of agents (up to 4), 
    or dispenser counts.
    
    Vector Layout (96 dimensions):
    - [0:14] Self Agent State (14 dims)
    - [14:56] Other Agents State (3 agents * 14 dims = 42 dims) (padded with 0 if not present)
    - [56:70] Recipe Orders HUD (Recipe 1 one-hot size 7 + Recipe 2 one-hot size 7 = 14 dims)
    - [70:76] Attentive Ingredient Dispensers (3 slots * 2 offsets [dx, dy] = 6 dims)
    - [76:96] K-Nearest Stations (5 types * 2 nearest * 2 offsets [dx, dy] = 20 dims):
      - 76:80: 2 nearest Chopping Boards
      - 80:84: 2 nearest Plate Dispensers
      - 84:88: 2 nearest Empty Counters
      - 88:92: 2 nearest Delivery Stations
      - 92:96: 2 nearest Trash Cans
    """
    
    def __init__(self, recipe_manager=None):
        self.recipe_manager = recipe_manager
        
    def _encode_item_type_onehot(self, item) -> np.ndarray:
        onehot = np.zeros(8, dtype=np.float32)
        if item is None:
            onehot[0] = 1.0
            return onehot
            
        name = item.get("name", "").lower()
        if name == "tomato":
            idx = 1
        elif name == "lettuce":
            idx = 2
        elif name == "onion":
            idx = 3
        elif name == "cucumber":
            idx = 4
        elif name == "plate":
            idx = 5
        elif "plate" in name or item.get("is_combined", False):
            idx = 6
        else:
            idx = 7
            
        onehot[idx] = 1.0
        return onehot

    def _encode_recipe_onehot(self, recipe_name: str) -> np.ndarray:
        onehot = np.zeros(7, dtype=np.float32)
        if not recipe_name:
            onehot[0] = 1.0
            return onehot
            
        name = recipe_name.upper()
        if name == "SIMPLE_SALAD":
            idx = 1
        elif name == "GARDEN_SALAD":
            idx = 2
        elif name == "FRESH_SALAD":
            idx = 3
        elif name == "ONION_SOUP":
            idx = 4
        elif name == "ONION_SALAD":
            idx = 5
        else:
            idx = 6
            
        onehot[idx] = 1.0
        return onehot

    def _get_recipe_ingredients(self, recipe_name: str, recipe_manager) -> List[str]:
        if not recipe_name or not recipe_manager:
            return []
        recipe = recipe_manager.get_recipe_by_name(recipe_name)
        if recipe:
            return [ing.lower() for ing in recipe.ingredients]
        return []

    def build_observations(
        self,
        game_state: Dict[str, Any],
        agent_ids: list,
        item_manager,
        recipe_manager=None
    ) -> Dict[str, np.ndarray]:
        
        map_size = game_state["map_static"].shape
        H, W = map_size
        recipe_mgr = recipe_manager or self.recipe_manager
        
        # 1. Parse all static locations
        dispensers_by_type = {}
        chopping_boards = []
        plate_dispensers = []
        empty_counters = []
        delivery_stations = []
        trash_cans = []
        
        # Parse dispensers from game state
        for (x, y), ing in game_state.get("dispensers", {}).items():
            ing_type = ing.lower()
            if ing_type not in dispensers_by_type:
                dispensers_by_type[ing_type] = []
            dispensers_by_type[ing_type].append((x, y))
            
        # Parse other stations from static grid
        for y in range(H):
            for x in range(W):
                cell_code = game_state["map_static"][y, x]
                cell_char = item_manager.get_cell_char(cell_code)
                
                if cell_char == 'C':
                    chopping_boards.append((x, y))
                elif cell_char == 'P':
                    plate_dispensers.append((x, y))
                elif cell_char == 'D':
                    delivery_stations.append((x, y))
                elif cell_char == 'T':
                    trash_cans.append((x, y))
                elif cell_char == '#':
                    if (x, y) not in game_state.get("ground_instances", {}):
                        empty_counters.append((x, y))
                        
        # 2. Get HUD recipe orders
        recipe_queue = game_state.get("recipe_order", [])
        rec_1 = recipe_queue[0] if len(recipe_queue) > 0 else None
        rec_2 = recipe_queue[1] if len(recipe_queue) > 1 else None
        
        recipe_features = np.concatenate([
            self._encode_recipe_onehot(rec_1),
            self._encode_recipe_onehot(rec_2)
        ])
        
        # Determine unique ingredients needed right now
        needed_ingredients = []
        for r_name in [rec_1, rec_2]:
            for ing in self._get_recipe_ingredients(r_name, recipe_mgr):
                if ing not in needed_ingredients:
                    needed_ingredients.append(ing)
        needed_ingredients = needed_ingredients[:3]  # Max 3 ingredient slots
        
        observations = {}
        
        for agent_id in agent_ids:
            # Self agent info
            pos = game_state["agent_pos"][agent_id]
            orient = game_state["agent_orientation"][agent_id]
            holding_id = game_state["agent_holding_instance"].get(agent_id, None)
            holding_item = game_state["item_instances"].get(holding_id) if holding_id is not None else None
            
            # Encode self state (14 dims)
            self_state = np.zeros(14, dtype=np.float32)
            self_state[0] = pos[0] / max(1, W - 1)
            self_state[1] = pos[1] / max(1, H - 1)
            self_state[2] = float(orient[0])
            self_state[3] = float(orient[1])
            self_state[4] = 1.0 if holding_item is not None else 0.0
            self_state[5:13] = self._encode_item_type_onehot(holding_item)
            if holding_item is not None and "cuts_completed" in holding_item and "cuts_needed" in holding_item:
                self_state[13] = holding_item["cuts_completed"] / max(1, holding_item["cuts_needed"])
            else:
                self_state[13] = 0.0
                
            # Encode other agents (up to 3, padded to 42 dims)
            other_agents = [a for a in agent_ids if a != agent_id][:3]
            other_state = np.zeros(42, dtype=np.float32)
            
            for o_idx, o_agent in enumerate(other_agents):
                o_pos = game_state["agent_pos"][o_agent]
                o_orient = game_state["agent_orientation"][o_agent]
                o_holding_id = game_state["agent_holding_instance"].get(o_agent, None)
                o_holding_item = game_state["item_instances"].get(o_holding_id) if o_holding_id is not None else None
                
                offset = o_idx * 14
                other_state[offset] = o_pos[0] / max(1, W - 1)
                other_state[offset+1] = o_pos[1] / max(1, H - 1)
                other_state[offset+2] = float(o_orient[0])
                other_state[offset+3] = float(o_orient[1])
                other_state[offset+4] = 1.0 if o_holding_item is not None else 0.0
                other_state[offset+5:offset+13] = self._encode_item_type_onehot(o_holding_item)
                if o_holding_item is not None and "cuts_completed" in o_holding_item and "cuts_needed" in o_holding_item:
                    other_state[offset+13] = o_holding_item["cuts_completed"] / max(1, o_holding_item["cuts_needed"])
                else:
                    other_state[offset+13] = 0.0
                    
            # 3. Calculate attentive ingredient offsets (6 dims)
            attentive_ingredients = np.zeros(6, dtype=np.float32)
            for ing_idx, ing_name in enumerate(needed_ingredients):
                dispensers = dispensers_by_type.get(ing_name, [])
                if dispensers:
                    closest = min(dispensers, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1]))
                    attentive_ingredients[ing_idx*2] = (closest[0] - pos[0]) / max(1, W - 1)
                    attentive_ingredients[ing_idx*2+1] = (closest[1] - pos[1]) / max(1, H - 1)
                    
            # 4. Calculate K-Nearest Stations (20 dims)
            station_features = np.zeros(20, dtype=np.float32)
            
            def populate_nearest_stations(stations: list, feature_offset: int):
                if not stations:
                    return
                # Sort by manhattan distance
                sorted_stations = sorted(stations, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1]))
                # Take top 2
                for s_idx, station in enumerate(sorted_stations[:2]):
                    dx = (station[0] - pos[0]) / max(1, W - 1)
                    dy = (station[1] - pos[1]) / max(1, H - 1)
                    station_features[feature_offset + s_idx*2] = dx
                    station_features[feature_offset + s_idx*2+1] = dy
                    
            populate_nearest_stations(chopping_boards, 0)
            populate_nearest_stations(plate_dispensers, 4)
            populate_nearest_stations(empty_counters, 8)
            populate_nearest_stations(delivery_stations, 12)
            populate_nearest_stations(trash_cans, 16)
            
            # Concatenate all features to build final 96-dim vector
            observations[agent_id] = np.concatenate([
                self_state,
                other_state,
                recipe_features,
                attentive_ingredients,
                station_features
            ]).astype(np.float32)
            
        return observations
