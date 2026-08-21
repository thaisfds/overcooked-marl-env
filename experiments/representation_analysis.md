# Representation Analysis: 3D Grid (CNN) vs. 1D Vector (MLP)

This document analyzes the scaling limits and performance trade-offs of both observation representation strategies in the Overcooked MARL environment as we scale:
1. **Number of Agents** (from 2 up to 4)
2. **Map Dimensions** (from 7x7 up to 30x30 or larger)
3. **Recipes & Ingredients** (scaling from 2 dispensers to 5+ dispensers)

---

## 📊 Comparison Matrix

| Feature | 3D Grid Representation (CNN) | 1D Vector Representation (MLP) |
| :--- | :--- | :--- |
| **Input Shape** | `(H, W, Channels)` (e.g., `7x7x9` or `15x11x9`) | `(N,)` (Flat array, e.g., `56` elements) |
| **Map Size Scaling** | **Quadratic ($O(H \times W)$)**. Scaling from 7x7 to 30x30 increases inputs from **441 to 8,100**. | **Constant ($O(1)$)**. Input size remains exactly `56` regardless of map size. |
| **Agent Scaling (up to 4)** | **Constant channels**. Positions are just placed as `1`s on the same static channels. | **Linear ($O(A)$)**. Adding 2 more agents adds $2 \times 14 = 28$ dimensions. |
| **Dispenser Scaling** | **Constant channels**. Dispensers are drawn on the static map channels. | **Linear ($O(D)$)** if absolute; **Constant ($O(1)$)** if context-based. |
| **Training Speed** | **Slow**. Convolutions require heavy backpropagation ($O(\text{kernel} \times \text{filters})$). | **Extremely Fast**. Linear dense layers require basic matrix multiplications. |
| **Generalization** | **Poor**. A CNN trained on a 7x7 map fails on a 15x15 map because spatial features shift. | **Excellent**. Agent learns relative offsets (`dx, dy`). It generalizes to any map size. |

---

## 🔍 How to Handle Scaling in the 1D Vector (No Dimension Explosion)

If we hardcode a distance feature for every single dispenser type (e.g., `nearest_tomato`, `nearest_lettuce`, `nearest_onion`, `nearest_cucumber`, etc.), the 1D vector size will grow linearly with every new ingredient added to the game. 

To prevent this dimension explosion, we can use **Contextual/Attentive Feature Extraction**:

### 1. Attentive Ingredient Offsets (Fixed Size)
Instead of list-based coordinates, the agent only gets relative offsets to the ingredients it **currently needs** to complete the active recipes in the HUD:
- **Ingredient 1 needed for active recipe:** `[dx, dy]` to nearest dispenser.
- **Ingredient 2 needed for active recipe:** `[dx, dy]` to nearest dispenser.
- **Ingredient 3 needed for active recipe:** `[dx, dy]` to nearest dispenser.
- *If the active recipe only needs 1 ingredient (e.g. lettuce salad), the other slots default to `[0.0, 0.0]`.*
- **Result:** Constant feature size of 6, regardless of whether the game has 3 or 30 different ingredient dispensers!

### 2. Attentive K-Nearest Stations
Instead of list-based coordinates for every chopping board or plate dispenser on the map, we only feed the top-$K$ nearest ones to the agent:
- **Nearest Chopping Board 1:** `[dx, dy, is_occupied]`
- **Nearest Chopping Board 2:** `[dx, dy, is_occupied]`
- **Nearest Plate Dispenser 1:** `[dx, dy]`
- **Result:** Keeps dimensions perfectly fixed, even on maps with 10 chopping boards and 5 plate dispensers.

### 3. Agent Padding (Fixed size for up to 4 agents)
We can fix the observation size to support **up to 4 agents** by allocating slots for 3 other agents:
- `Other Agent 1 State`: 14 dims
- `Other Agent 2 State`: 14 dims
- `Other Agent 3 State`: 14 dims
- *If the map only has 2 agents (like our tutorial map), Agent 2 and Agent 3 slots are simply padded with `0.0`.*
- **Result:** The policy neural network shape remains 100% identical, allowing the same model to play on 1-agent, 2-agent, or 4-agent maps!

---

## 💡 Recommendation

1. **Keep the 1D MLP approach:** It is vastly superior in training speed (11 seconds vs 5.5 minutes) and generalizes much better to different map shapes.
2. **Adopt Attentive & Padded Features:**
   By padding the "other agents" space to a max of 3 other agents, and using relative offsets only for active/needed ingredients, we get a **completely fixed 1D vector size** (e.g., ~80 dimensions) that:
   - Supports 1 to 4 agents.
   - Supports any map size.
   - Supports any number of recipes and dispensers.
