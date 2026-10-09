import numpy as np
from typing import Dict, Tuple, Any, List, Optional
from ..core.items import ItemType, ItemState, is_ready
from ..core.constants import MAX_EPISODE_STEPS

class FeaturizedObservationBuilder:
    """
    Builds a dense, normalized, scale-invariant 1D vector representation of the game state.
    Vector shape is ALWAYS exactly (150,) regardless of map size, number of agents (up to 4),
    recipes, ingredients, or counter count.
    
    Vector Layout (150 dimensions):
    - [0:24]   Self Agent State (24 dims)
    - [24:32]  Facing Tile Vision (8 dims)
    - [32:80]  Other Agents State (3 partners * 16 dims = 48 dims) (zero-padded if fewer agents)
    - [80:100] HUD Recipe Orders (Goal-Conditioned) (2 recipes * 10 dims = 20 dims)
    - [100:108] Attentive Ingredient Dispensers (4 slots * 2 [dx, dy] = 8 dims)
    - [108:132] Fixed Stations & Universal Processing Appliances (24 dims):
                - 108:112: 2 nearest Delivery Stations (4 dims)
                - 112:116: 2 nearest Plate Dispensers (4 dims)
                - 116:118: 1 nearest Trash Can (2 dims)
                - 118:130: 2 nearest Universal Processing Appliances (2 * 6 = 12 dims)
                - 130:132: Goal-Directed Appliance Beacon (2 dims)
    - [132:150] Semantic Counters & Multi-Agent Transfer (18 dims):
                - 132:136: 2 nearest Empty Counters (4 dims)
                - 136:140: 2 nearest Counters with Plates/Dishes (4 dims)
                - 140:144: 2 nearest Counters with Ready Ingredients (4 dims)
                - 144:148: 2 nearest Transfer / Pass-Through Counters (4 dims)
                - 148:150: Transfer Counters Occupancy / Content Flags (2 dims)
    """

    def __init__(self, recipe_manager=None):
        self.recipe_manager = recipe_manager

    def _get_recipe_ingredients(self, recipe_name: str, recipe_manager=None) -> List[str]:
        if not recipe_name:
            return []
        recipe_mgr = recipe_manager or self.recipe_manager
        if recipe_mgr:
            recipe = recipe_mgr.get_recipe_by_name(recipe_name)
            if recipe and hasattr(recipe, "ingredients") and recipe.ingredients:
                return [ing.lower() for ing in recipe.ingredients]
        return []

    def _get_pending_recipes(self, game_state: Dict[str, Any]) -> List[str]:
        recipe_queue = list(game_state.get("recipe_order", []))
        for delivered in game_state.get("delivered_recipes", []):
            if delivered in recipe_queue:
                recipe_queue.remove(delivered)
        return recipe_queue

    def _build_map_ingredient_catalog(self, game_state: Dict[str, Any], recipe_manager=None) -> List[str]:
        """Builds a deterministic, sorted catalog of up to 8 unique ingredients present in this map/recipes."""
        catalog_set = set()
        for ing in game_state.get("dispensers", {}).values():
            if isinstance(ing, str):
                catalog_set.add(ing.strip().lower())
        recipe_mgr = recipe_manager or self.recipe_manager
        if recipe_mgr and hasattr(recipe_mgr, "recipes"):
            for r in recipe_mgr.recipes:
                if hasattr(r, "ingredients"):
                    for ing in r.ingredients:
                        catalog_set.add(ing.strip().lower())
        return sorted(list(catalog_set))[:8]

    def _get_item_category_onehot(self, item: Optional[Dict]) -> np.ndarray:
        """Returns 4-dim one-hot: [None, Raw Ingredient, Ready Ingredient, Dish]."""
        cat = np.zeros(4, dtype=np.float32)
        if item is None:
            cat[0] = 1.0
            return cat
        itype = item.get("type")
        if itype == ItemType.DISH or "plate" in item.get("name", "").lower():
            cat[3] = 1.0
        elif is_ready(item):
            cat[2] = 1.0
        else:
            cat[1] = 1.0
        return cat

    def _get_item_ingredients_multihot(self, item: Optional[Dict], ing_to_slot: Dict[str, int]) -> np.ndarray:
        """Returns 8-dim multi-hot of ingredients present in the item (or plate)."""
        multihot = np.zeros(8, dtype=np.float32)
        if not item or not ing_to_slot:
            return multihot
        itype = item.get("type")
        if itype == ItemType.DISH or "plate" in item.get("name", "").lower():
            for ing in item.get("ingredients", []):
                slot = ing_to_slot.get(ing.lower())
                if slot is not None and 0 <= slot < 8:
                    multihot[slot] = 1.0
        else:
            name = item.get("name", "").lower()
            slot = ing_to_slot.get(name)
            if slot is not None and 0 <= slot < 8:
                multihot[slot] = 1.0
        return multihot

    def build_observations(
        self,
        game_state: Dict[str, Any],
        agent_ids: list,
        item_manager,
        recipe_manager=None
    ) -> Dict[str, np.ndarray]:
        
        map_static = game_state["map_static"]
        H, W = map_static.shape
        recipe_mgr = recipe_manager or self.recipe_manager
        total_steps_limit = getattr(game_state, "max_steps", MAX_EPISODE_STEPS) or MAX_EPISODE_STEPS

        # 1. Map Ingredient Catalog & Slot Mapping (up to 8 slots)
        map_catalog = self._build_map_ingredient_catalog(game_state, recipe_mgr)
        ing_to_slot = {ing: i for i, ing in enumerate(map_catalog)}

        # Dispensers mapping
        dispensers_by_type: Dict[str, List[Tuple[int, int]]] = {}
        for (x, y), ing in game_state.get("dispensers", {}).items():
            ing_type = ing.lower()
            dispensers_by_type.setdefault(ing_type, []).append((x, y))

        # 2. Parse Stations & Counters
        delivery_stations: List[Tuple[int, int]] = []
        plate_dispensers: List[Tuple[int, int]] = []
        trash_cans: List[Tuple[int, int]] = []
        processing_appliances: List[Tuple[int, int, float]] = []  # (x, y, station_code)
        empty_counters: List[Tuple[int, int]] = []
        counters_with_dishes: List[Tuple[int, int]] = []
        counters_with_ready_ing: List[Tuple[int, int]] = []
        all_counters: List[Tuple[int, int]] = []

        ground_instances = game_state.get("ground_instances", {})
        item_instances = game_state.get("item_instances", {})

        for y in range(H):
            for x in range(W):
                code = map_static[y, x]
                char = item_manager.get_cell_char(code) if item_manager else ' '

                if char == 'D':
                    delivery_stations.append((x, y))
                elif char == 'P':
                    plate_dispensers.append((x, y))
                elif char == 'T':
                    trash_cans.append((x, y))
                elif char == '#':
                    all_counters.append((x, y))
                    iid = ground_instances.get((x, y))
                    if iid is None:
                        empty_counters.append((x, y))
                    else:
                        g_item = item_instances.get(iid)
                        if g_item:
                            if g_item.get("type") == ItemType.DISH or "plate" in g_item.get("name", "").lower():
                                counters_with_dishes.append((x, y))
                            elif is_ready(g_item):
                                counters_with_ready_ing.append((x, y))
                elif char not in (' ', '|', '_', 'I'):
                    # Any interactive processing appliance (C=Chop, S=Stove, W=Sink, F=Fryer, O=Oven...)
                    norm_code = float(code) / 10.0
                    processing_appliances.append((x, y, norm_code))

        # 3. Dynamic Midpoint for Multi-Agent Transfer Counters
        agent_positions = [game_state["agent_pos"][a] for a in agent_ids if a in game_state.get("agent_pos", {})]
        if agent_positions:
            mid_x = float(np.mean([p[0] for p in agent_positions]))
            mid_y = float(np.mean([p[1] for p in agent_positions]))
        else:
            mid_x, mid_y = W / 2.0, H / 2.0

        # Sort all counters by distance to agents midpoint to identify transfer zone
        transfer_counters = sorted(all_counters, key=lambda c: abs(c[0] - mid_x) + abs(c[1] - mid_y))[:2]

        # 4. Parse Pending Recipes (Goal-Conditioned HUD)
        pending_recipes = self._get_pending_recipes(game_state)
        rec_1 = pending_recipes[0] if len(pending_recipes) > 0 else None
        rec_2 = pending_recipes[1] if len(pending_recipes) > 1 else None

        rec1_req = self._get_recipe_ingredients(rec_1, recipe_mgr)
        rec2_req = self._get_recipe_ingredients(rec_2, recipe_mgr)

        delivered_recipes = game_state.get("delivered_recipes", [])
        total_recipes_count = len(game_state.get("recipe_order", []))
        dishes_delivered_ratio = len(delivered_recipes) / max(1, total_recipes_count) if total_recipes_count > 0 else 0.0
        episode_time_ratio = float(game_state.get("time_steps", 0)) / max(1, total_steps_limit)

        # Build HUD features (20 dims)
        hud_recipe_state = np.zeros(20, dtype=np.float32)
        # Recipe 1 (10 dims: total_req, remaining_req, 8-dim multihot)
        if rec_1:
            hud_recipe_state[0] = len(rec1_req) / 4.0
            hud_recipe_state[1] = len(rec1_req) / 4.0
            for ing in rec1_req:
                s = ing_to_slot.get(ing)
                if s is not None and 0 <= s < 8:
                    hud_recipe_state[2 + s] = 1.0

        # Recipe 2 (10 dims: total_req, remaining_req, 8-dim multihot)
        if rec_2:
            hud_recipe_state[10] = len(rec2_req) / 4.0
            hud_recipe_state[11] = len(rec2_req) / 4.0
            for ing in rec2_req:
                s = ing_to_slot.get(ing)
                if s is not None and 0 <= s < 8:
                    hud_recipe_state[12 + s] = 1.0

        observations: Dict[str, np.ndarray] = {}

        # 5. Build Agent-Specific Observations
        for agent_id in agent_ids:
            pos = game_state["agent_pos"][agent_id]
            orient = game_state["agent_orientation"][agent_id]
            holding_id = game_state["agent_holding_instance"].get(agent_id, None)
            holding_item = item_instances.get(holding_id) if holding_id is not None else None

            # --- BLOCO 1: Estado Próprio do Agente (24 dims) ---
            self_state = np.zeros(24, dtype=np.float32)
            self_state[0] = pos[0] / max(1, W - 1)
            self_state[1] = pos[1] / max(1, H - 1)
            self_state[2] = float(orient[0])
            self_state[3] = float(orient[1])
            self_state[4] = 1.0 if holding_item is not None else 0.0
            self_state[5:9] = self._get_item_category_onehot(holding_item)
            self_state[9:17] = self._get_item_ingredients_multihot(holding_item, ing_to_slot)

            cuts_progress = 0.0
            if holding_item is not None and "cuts_completed" in holding_item and "cuts_needed" in holding_item:
                cuts_progress = float(holding_item["cuts_completed"]) / max(1, float(holding_item["cuts_needed"]))
            self_state[17] = cuts_progress
            self_state[18] = 1.0 if (holding_item is not None and is_ready(holding_item)) else 0.0

            # Dish Target Alignment
            is_holding_target = 0.0
            is_holding_invalid = 0.0
            dish_ratio = 0.0
            if holding_item is not None and (holding_item.get("type") == ItemType.DISH or "plate" in holding_item.get("name", "").lower()):
                plate_ings = [i.lower() for i in holding_item.get("ingredients", [])]
                if rec_1:
                    dish_ratio = len(plate_ings) / max(1, len(rec1_req))
                    if sorted(plate_ings) == sorted(rec1_req):
                        is_holding_target = 1.0
                    else:
                        # Check if has any ingredient not in any pending recipe
                        all_pending_reqs = set()
                        for pr in pending_recipes:
                            all_pending_reqs.update(self._get_recipe_ingredients(pr, recipe_mgr))
                        if any(pi not in all_pending_reqs for pi in plate_ings):
                            is_holding_invalid = 1.0
            self_state[19] = is_holding_target
            self_state[20] = is_holding_invalid
            self_state[21] = dish_ratio
            self_state[22] = episode_time_ratio
            self_state[23] = dishes_delivered_ratio

            # --- BLOCO 2: Visão do Tile à Frente (8 dims) ---
            facing_state = np.zeros(8, dtype=np.float32)
            front_tile = (pos[0] + orient[0], pos[1] + orient[1])
            if 0 <= front_tile[0] < W and 0 <= front_tile[1] < H:
                f_code = map_static[front_tile[1], front_tile[0]]
                f_char = item_manager.get_cell_char(f_code) if item_manager else ' '
                if f_char == '#':
                    f_station_norm = 0.2
                elif f_char in ('C', 'S', 'W', 'F', 'O'):
                    f_station_norm = 0.4
                elif f_char == 'I':
                    f_station_norm = 0.6
                elif f_char == 'D':
                    f_station_norm = 0.8
                elif f_char in ('P', 'T'):
                    f_station_norm = 1.0
                else:
                    f_station_norm = 0.0
                facing_state[0] = f_station_norm

                f_iid = ground_instances.get(front_tile)
                f_item = item_instances.get(f_iid) if f_iid is not None else None
                if f_item is not None:
                    facing_state[1] = 1.0
                    is_f_dish = f_item.get("type") == ItemType.DISH or "plate" in f_item.get("name", "").lower()
                    facing_state[2] = 1.0 if is_f_dish else 0.0
                    if is_f_dish and rec_1:
                        f_plate_ings = [i.lower() for i in f_item.get("ingredients", [])]
                        if sorted(f_plate_ings) == sorted(rec1_req):
                            facing_state[3] = 1.0
                    elif not is_f_dish:
                        facing_state[4] = 1.0 if is_ready(f_item) else 0.0
                        if rec_1 and f_item.get("name", "").lower() in rec1_req:
                            facing_state[5] = 1.0
                    if "cuts_completed" in f_item and "cuts_needed" in f_item:
                        facing_state[6] = float(f_item["cuts_completed"]) / max(1, float(f_item["cuts_needed"]))
                    if is_f_dish:
                        facing_state[7] = 1.0 if f_item.get("ingredients") else 0.75
                    else:
                        facing_state[7] = 0.5 if is_ready(f_item) else 0.25

            # --- BLOCO 3: Outros Agentes (Até 3 parceiros * 16 dims = 48 dims) ---
            other_state = np.zeros(48, dtype=np.float32)
            other_agents = [a for a in agent_ids if a != agent_id][:3]
            for o_idx, o_agent in enumerate(other_agents):
                offset = o_idx * 16
                o_pos = game_state["agent_pos"].get(o_agent)
                if o_pos is None:
                    continue
                o_orient = game_state["agent_orientation"].get(o_agent, (0, 0))
                o_hid = game_state["agent_holding_instance"].get(o_agent, None)
                o_item = item_instances.get(o_hid) if o_hid is not None else None

                dx = (o_pos[0] - pos[0]) / max(1, W - 1)
                dy = (o_pos[1] - pos[1]) / max(1, H - 1)
                dist_norm = (abs(o_pos[0] - pos[0]) + abs(o_pos[1] - pos[1])) / max(1, W + H - 2)

                other_state[offset + 0] = dx
                other_state[offset + 1] = dy
                other_state[offset + 2] = dist_norm
                other_state[offset + 3] = float(o_orient[0])
                other_state[offset + 4] = float(o_orient[1])
                other_state[offset + 5] = 1.0 if o_item is not None else 0.0
                other_state[offset + 6:offset + 10] = self._get_item_category_onehot(o_item)
                
                o_holding_ready_dish = 0.0
                o_holds_needed = 0.0
                if o_item is not None:
                    if (o_item.get("type") == ItemType.DISH or "plate" in o_item.get("name", "").lower()) and rec_1:
                        o_ings = [i.lower() for i in o_item.get("ingredients", [])]
                        if sorted(o_ings) == sorted(rec1_req):
                            o_holding_ready_dish = 1.0
                    elif rec_1 and o_item.get("name", "").lower() in rec1_req:
                        o_holds_needed = 1.0
                other_state[offset + 10] = o_holding_ready_dish

                # Partner facing station
                o_ftile = (o_pos[0] + o_orient[0], o_pos[1] + o_orient[1])
                if 0 <= o_ftile[0] < W and 0 <= o_ftile[1] < H:
                    other_state[offset + 11] = float(map_static[o_ftile[1], o_ftile[0]]) / 10.0
                other_state[offset + 12] = 1.0  # partner is active
                if o_item and "cuts_completed" in o_item and "cuts_needed" in o_item:
                    other_state[offset + 13] = float(o_item["cuts_completed"]) / max(1, float(o_item["cuts_needed"]))
                other_state[offset + 14] = o_holds_needed
                if transfer_counters:
                    tc_dist = min(abs(o_pos[0] - tc[0]) + abs(o_pos[1] - tc[1]) for tc in transfer_counters)
                    other_state[offset + 15] = tc_dist / max(1, W + H - 2)

            # --- BLOCO 5: Attentive Ingredient Dispensers (8 dims) ---
            # Identifies dispensers for up to 4 ingredients still needed by recipe 1
            attentive_dispensers = np.zeros(8, dtype=np.float32)
            needed_ings_rec1 = []
            if rec_1:
                # If holding a dish, find what ingredients are still missing
                held_ings = [i.lower() for i in holding_item.get("ingredients", [])] if (holding_item and (holding_item.get("type") == ItemType.DISH or "plate" in holding_item.get("name", "").lower())) else []
                missing = [ing for ing in rec1_req if ing not in held_ings]
                needed_ings_rec1 = missing if missing else rec1_req
            for ing_idx, ing_name in enumerate(needed_ings_rec1[:4]):
                disp_list = dispensers_by_type.get(ing_name, [])
                if disp_list:
                    closest = min(disp_list, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
                    attentive_dispensers[ing_idx * 2] = (closest[0] - pos[0]) / max(1, W - 1)
                    attentive_dispensers[ing_idx * 2 + 1] = (closest[1] - pos[1]) / max(1, H - 1)

            # --- BLOCO 6: Estações Estáticas & Eletrodomésticos Universais (24 dims) ---
            appliance_and_flow_state = np.zeros(24, dtype=np.float32)
            
            # Helper to populate relative offsets of nearest stations
            def set_nearest(targets: List[Tuple[int, int]], out_arr: np.ndarray, start_idx: int, count: int):
                if not targets:
                    return
                sorted_t = sorted(targets, key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))
                for idx, t in enumerate(sorted_t[:count]):
                    out_arr[start_idx + idx * 2] = (t[0] - pos[0]) / max(1, W - 1)
                    out_arr[start_idx + idx * 2 + 1] = (t[1] - pos[1]) / max(1, H - 1)

            set_nearest(delivery_stations, appliance_and_flow_state, 0, 2)     # 4 dims (0:4)
            set_nearest(plate_dispensers, appliance_and_flow_state, 4, 2)       # 4 dims (4:8)
            set_nearest(trash_cans, appliance_and_flow_state, 8, 1)             # 2 dims (8:10)

            # 2 nearest Universal Processing Appliances (12 dims: 10:22)
            if processing_appliances:
                sorted_app = sorted(processing_appliances, key=lambda a: abs(a[0] - pos[0]) + abs(a[1] - pos[1]))
                for a_idx, app in enumerate(sorted_app[:2]):
                    base_a = 10 + a_idx * 6
                    appliance_and_flow_state[base_a] = (app[0] - pos[0]) / max(1, W - 1)
                    appliance_and_flow_state[base_a + 1] = (app[1] - pos[1]) / max(1, H - 1)
                    appliance_and_flow_state[base_a + 2] = app[2]  # normalized station code
                    
                    app_iid = ground_instances.get((app[0], app[1]))
                    app_item = item_instances.get(app_iid) if app_iid is not None else None
                    if app_item is not None:
                        appliance_and_flow_state[base_a + 3] = 1.0  # is occupied
                        appliance_and_flow_state[base_a + 4] = 1.0 if is_ready(app_item) else 0.0
                        if "cuts_completed" in app_item and "cuts_needed" in app_item:
                            appliance_and_flow_state[base_a + 5] = float(app_item["cuts_completed"]) / max(1, float(app_item["cuts_needed"]))

            # Goal-Directed Appliance Beacon (2 dims: 22:24)
            beacon_target = None
            if is_holding_target and delivery_stations:
                beacon_target = min(delivery_stations, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
            elif holding_item and not is_ready(holding_item) and processing_appliances:
                # Holding raw ingredient: point to closest available processing appliance
                beacon_target = min(processing_appliances, key=lambda a: abs(a[0] - pos[0]) + abs(a[1] - pos[1]))
            elif holding_item and (holding_item.get("type") == ItemType.DISH or "plate" in holding_item.get("name", "").lower()):
                # Holding plate: point to closest appliance with ready food, or ready counter
                ready_sources = []
                for app in processing_appliances:
                    gid = ground_instances.get((app[0], app[1]))
                    if gid is not None:
                        gitem = item_instances.get(gid)
                        if gitem is not None and is_ready(gitem):
                            ready_sources.append(app)
                if ready_sources:
                    beacon_target = min(ready_sources, key=lambda a: abs(a[0] - pos[0]) + abs(a[1] - pos[1]))
            if beacon_target is not None:
                appliance_and_flow_state[22] = (beacon_target[0] - pos[0]) / max(1, W - 1)
                appliance_and_flow_state[23] = (beacon_target[1] - pos[1]) / max(1, H - 1)

            # --- BLOCO 7: Bancadas Semânticas & Transferência Multiagente (18 dims) ---
            semantic_counters_state = np.zeros(18, dtype=np.float32)
            set_nearest(empty_counters, semantic_counters_state, 0, 2)           # 4 dims (0:4)
            set_nearest(counters_with_dishes, semantic_counters_state, 4, 2)     # 4 dims (4:8)
            set_nearest(counters_with_ready_ing, semantic_counters_state, 8, 2)  # 4 dims (8:12)
            set_nearest(transfer_counters, semantic_counters_state, 12, 2)       # 4 dims (12:16)

            # Transfer counter occupancy flags (2 dims: 16:18)
            if len(transfer_counters) > 0 and transfer_counters[0] in ground_instances:
                semantic_counters_state[16] = 1.0
            if len(transfer_counters) > 1 and transfer_counters[1] in ground_instances:
                semantic_counters_state[17] = 1.0

            # --- CONCATENAÇÃO FINAL (ESTRITAMENTE 150 DIMENSÕES) ---
            final_vec = np.concatenate([
                self_state,                # 24 dims
                facing_state,              # 8 dims
                other_state,               # 48 dims
                hud_recipe_state,          # 20 dims
                attentive_dispensers,      # 8 dims
                appliance_and_flow_state,  # 24 dims
                semantic_counters_state    # 18 dims
            ]).astype(np.float32)

            assert len(final_vec) == 150, f"Vector size mismatch! Expected 150, got {len(final_vec)}"
            observations[agent_id] = final_vec

        return observations
