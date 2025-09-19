# Protoss Natural Wall Planner (Map-Agnostic)

## Goal
Compute a **map-agnostic natural wall** for Protoss every game (no templates), then place 2 Gateways/Core + a controlled gap and a rear Pylon. It must work on new ladder maps without framework updates.

### Goal pattern:

Default is Gateway → Cybernetics Core → Gateway (3×3 chain across the natural choke).

Always leave a 1-tile seam for the Gatekeeper Zealot to hold position.

### Supporting details:

Pylons: Positioned behind the wall (toward the natural Nexus) to power the wall buildings safely. Avoid putting them at the very front.

Shield Battery space: Reserve a tile or two just behind the seam for a Shield Battery (and Cannon if needed).

Orientation: Defensive bias (builds hug the natural side of the choke, not pushed out toward the map center).

### Choke width rules:

Narrow (≤7 tiles): 1× Gateway + Gatekeeper Zealot can suffice.

Normal (8–11 tiles): Gate–Core–Gate is the standard.

Wide (≥12 tiles): Add a 3rd Gateway or other 3×3 if needed; if still unwallable, mark as fallback and just rely on Gatekeeper + batteries.

## Inputs (from python-sc2 / ARES)
- `game_info.placement_grid` → buildable tiles (numpy `H×W`, 1/0).
- `game_info.pathing_grid`   → walkable tiles (numpy `H×W`, 1/0).
- `game_info.map_center`     → `Point2`.
- `mediator.get_own_nat`     → natural expansion `Point2`.

## Outputs
- `WallResult`:
  - `buildings`: list of `(name, (tx, ty))` lower-left tile coords for 3×3 buildings in order: Gateway, Cybernetics Core, Forge.
  - `pylon`: `(tx, ty)` for a 2×2 rear Pylon, or `None`.
  - `gap_tile`: `(tx, ty)` for the one-tile gap, or `None` if hard wall.

## High-level Method
1. **Window extraction**  
   Cut a ~25×25 tile window centered a few tiles from the natural toward map center. From this window, read:
   - `placement_subgrid` (buildable),
   - `pathing_subgrid` (walkable),
   - and remember `(x0, y0)` to convert back to global tiles.

2. **Choke detection**
   - Collect walkable pixels in the window.
   - Compute **principal axis (PCA)** to get the **tangent** direction of the choke; `normal` is perpendicular.
   - Slice the walkable set along the **normal** and pick the slice with **minimum width** along the **tangent** → that slice’s centroid is **`choke_center`**.

3. **KD-tree over buildable tiles**
   - Build a KD-tree (or fallback spatial index) of **buildable tile centers** within the window.
   - Provide `nearest(x, y)` → returns the nearest buildable tile center.

4. **3×3 chain fitting**
   - Generate **candidate anchors** along the choke line: `choke_center + k * tangent * 3` for `k ∈ [-3..3]`.
   - For each anchor, lay out three 3×3 footprints (Gateway → Core → Forge) **collinear** along the tangent.
   - **Snap** each intended 3×3 **center** to nearest legal tile via KD-tree, convert center → lower-left `(tx, ty)`, and verify all tiles are buildable.

5. **Validate the wall**
   - Copy the walkable grid, **block** all 3×3 footprints.
   - If target gap is 1 tile, **carve** exactly one tile between two adjacent buildings (choose seam closer to the natural).
   - **Dilate** the walkable grid by the intruder radius (zergling ≈ 0.5 tiles).
   - **Flood-fill** from “outside” to “inside”:
     - If `gap_width=0`: no path.
     - If `gap_width=1`: path exists and passes through the carved gap.

6. **Rear Pylon placement**
   - Take centroid of 3×3s, step **opposite the normal** by ~3 tiles (toward natural), snap to nearest legal tile (KD-tree), verify 2×2 fits.

7. **Cache per map**
   - Store `WallResult` keyed by `map_name`.

## Public API
```python
result = await wall_planner.ensure_nat_wall(bot)
# → WallResult or None

for name, (tx, ty) in result.buildings:
    world = bot.game_info.grid_to_world((tx + 1.5, ty + 1.5))
    build(name, world)

if result.pylon:
    px, py = result.pylon
    build("PYLON", bot.game_info.grid_to_world((px + 1.0, py + 1.0)))
```

## Data Structures
```python
@dataclass
class BuildingSpec:
    name: str
    size: (int,int)

@dataclass
class WallSpec:
    three_by_threes: list[BuildingSpec]
    gap_width: int
    intruder_radius_tiles: float

@dataclass
class WallResult:
    buildings: list[tuple[str, tuple[int,int]]]
    pylon: tuple[int,int] | None
    gap_tile: tuple[int,int] | None
```

## Acceptance Criteria
- Works first game on unseen maps in <30 ms.
- PvZ Zealot gap → only path through gap.
- PvT hard wall → no path after dilation.
- Pylon behind wall powers 3×3s.
- If invalid, retries larger window or returns partial wall.

## Config
- `window half-size` (default 12).
- `gap_width` (0 vs T, 1 vs Z).
- `dilation kernel` size (3×3 default).

## Failure & Fallback
- Expand window if no valid chain.
- If still failing, return partial wall.

## Notes
- KD-tree is just speedup for snapping.
- Geometry is local, no templates.
- Validate with **connectivity on dilated grid**.
