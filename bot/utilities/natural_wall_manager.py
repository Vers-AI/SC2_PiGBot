"""
Purpose: Generate map-agnostic Protoss natural wall placements dynamically
Key Decisions: Uses SC2MapAnalysis for spatial operations, integrates with ARES via YAML generation
Limitations: Currently only supports Protoss natural walls, requires on_start initialization
"""

import yaml
import os
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sc2.position import Point2
from sc2.race import Race
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

@dataclass
class ChokeData:
    """Data about the detected choke point"""
    center: Point2
    width: float
    tangent: Point2  # Direction along the choke line
    normal: Point2   # Direction perpendicular to choke (toward natural)

@dataclass 
class WallPositions:
    """Calculated wall positions in world coordinates"""
    first_pylon: List[List[float]]
    pylons: List[List[float]]  
    three_by_threes: List[List[float]]
    static_defences: List[List[float]]
    gate_keeper: List[List[float]]

class NaturalWallManager:
    """
    Generates map-agnostic natural wall placements for new maps and integrates with ARES
    via building_placements.yml generation.
    
    Default pattern: Gateway → Cybernetics Core → Gateway with 1-tile zealot gap.
    Adapts to narrow (2 buildings) or wide chokes (3+ buildings) automatically.
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.ares_yml_path = Path("ares-sc2/src/ares/building_placements.yml")
        self.custom_yml_path = Path("building_placements.yml")  # Bot root
        self.generated_maps = set()  # Cache to avoid regeneration
        
    async def ensure_map_wall_exists(self, opponent_race: str) -> bool:
        """
        Check if current map has wall data for the given race, generate if missing.
        
        Args:
            opponent_race: "Zerg", "Terran", "Protoss", or "Random"
            
        Returns:
            True if wall data exists or was successfully generated
        """
        map_name = self.bot.game_info.map_name
        cache_key = f"{map_name}_{opponent_race}"
        
        # Check cache first
        if cache_key in self.generated_maps:
            return True
            
        # Check if map already exists in either file
        if self.map_has_wall_data(map_name, opponent_race):
            self.generated_maps.add(cache_key)
            return True
            
        logger.info(f"Generating wall placement for {map_name} vs {opponent_race}")
        
        try:
            # Generate new wall placement for this map
            wall_positions = await self.generate_wall_placement(opponent_race)
            
            if wall_positions is None:
                logger.warning(f"Failed to generate wall for {map_name} vs {opponent_race}")
                return False
                
            # Update custom YAML file
            self.update_custom_yaml(map_name, opponent_race, wall_positions)
            self.generated_maps.add(cache_key)
            
            logger.info(f"Successfully generated wall placement for {map_name} vs {opponent_race}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating wall placement for {map_name}: {e}")
            return False
        
    def map_has_wall_data(self, map_name: str, race: str) -> bool:
        """Check if map+race combo exists in ARES or custom YAML"""
        # Check ARES default file
        if self.ares_yml_path.exists():
            try:
                ares_data = self.load_yaml(self.ares_yml_path)
                if self.check_map_exists(ares_data, map_name, race):
                    logger.debug(f"Found existing wall data in ARES for {map_name} vs {race}")
                    return True
            except Exception as e:
                logger.warning(f"Error reading ARES YAML: {e}")
                
        # Check custom file 
        if self.custom_yml_path.exists():
            try:
                custom_data = self.load_yaml(self.custom_yml_path)
                if self.check_map_exists(custom_data, map_name, race):
                    logger.debug(f"Found existing wall data in custom file for {map_name} vs {race}")
                    return True
            except Exception as e:
                logger.warning(f"Error reading custom YAML: {e}")
                
        return False
        
    def check_map_exists(self, yaml_data: dict, map_name: str, race: str) -> bool:
        """Check if specific map+race exists in YAML data"""
        try:
            protoss_data = yaml_data.get("Protoss", {})
            map_data = protoss_data.get(map_name, {})
            wall_type = f"Vs{race}NatWall"
            return wall_type in map_data
        except:
            return False
            
    def load_yaml(self, file_path: Path) -> dict:
        """Load YAML file safely"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML from {file_path}: {e}")
            return {}
            
    def save_yaml(self, data: dict, file_path: Path):
        """Save YAML file safely"""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Failed to save YAML to {file_path}: {e}")
            raise
    
    async def generate_wall_placement(self, opponent_race: str) -> Optional[WallPositions]:
        """
        Generate wall placement using choke detection algorithm + SC2MapAnalysis.
        
        This is the main algorithm implementation that will be expanded.
        """
        try:
            # Get SC2MapAnalysis access
            map_data = self.bot.mediator.get_map_data_object
            
            # 1. Find the actual enemy approach path
            enemy_start = self.bot.enemy_start_locations[0]
            natural_pos = self.bot.mediator.get_own_nat
            
            # Use ARES pathfinding to get the real path enemies will take
            path = self.bot.mediator.find_raw_path(
                start=enemy_start,
                target=natural_pos,
                grid=map_data.get_pyastar_grid()
            )
            
            if not path or len(path) < 3:
                logger.warning("Could not find valid path from enemy to natural")
                return None
                
            # 2. Detect the narrowest choke along this path
            choke_data = self.find_choke_along_path(path, map_data)
            if choke_data is None:
                logger.warning("Could not detect choke along enemy path")
                return None
                
            # 3. Determine building layout based on choke width and race
            buildings = self.determine_building_layout(choke_data.width, opponent_race)
            
            # 4. Calculate actual wall positions using SC2MapAnalysis
            wall_positions = await self.calculate_wall_positions(choke_data, buildings, map_data)
            
            if wall_positions is None:
                logger.warning("Could not calculate valid wall positions")
                return None
                
            # 5. Validate wall connectivity (will implement later)
            if not await self.validate_wall_connectivity(wall_positions, opponent_race):
                logger.warning("Generated wall failed connectivity validation")
                # TODO: Implement fallback generation
                return None
                
            return wall_positions
            
        except Exception as e:
            logger.error(f"Error in generate_wall_placement: {e}")
            return None
    
    def find_choke_along_path(self, path: List[Point2], map_data) -> Optional[ChokeData]:
        """
        Find the narrowest cross-section along the enemy approach path.
        
        Implements PCA-based choke detection:
        1. Extract walkable area around the path
        2. Use PCA to find principal axis (tangent direction)  
        3. Slice along normal to find narrowest width
        """
        try:
            if len(path) < 3:
                logger.warning("Path too short for choke detection")
                return None
                
            # Get clean pathing grid
            pathing_grid = map_data.get_pyastar_grid()
            
            # Extract window around the path midpoint
            window_size = 25  # 25x25 tile window
            path_center = path[len(path) // 2]
            window_bounds = self.extract_analysis_window(path_center, window_size, pathing_grid.shape)
            
            if window_bounds is None:
                logger.warning("Could not extract analysis window")
                return None
                
            # Extract walkable pixels in window
            walkable_pixels = self.get_walkable_pixels_in_window(pathing_grid, window_bounds)
            
            if len(walkable_pixels) < 10:
                logger.warning("Not enough walkable pixels for PCA analysis")
                return None
                
            # Apply PCA to find principal axis
            pca_result = self.apply_pca_to_walkable_area(walkable_pixels)
            if pca_result is None:
                return None
                
            tangent, normal = pca_result
            
            # Find narrowest slice along the normal direction
            choke_data = self.find_narrowest_slice(walkable_pixels, tangent, normal, window_bounds)
            
            if choke_data is None:
                logger.warning("Could not find valid choke in walkable area")
                return None
                
            return choke_data
            
        except Exception as e:
            logger.error(f"Error in choke detection: {e}")
            return None
            
    def extract_analysis_window(self, center: Point2, window_size: int, grid_shape: tuple) -> Optional[Dict]:
        """Extract window bounds around center point"""
        half_size = window_size // 2
        
        # Calculate window boundaries
        x_min = max(0, int(center.x - half_size))
        x_max = min(grid_shape[0], int(center.x + half_size))
        y_min = max(0, int(center.y - half_size))
        y_max = min(grid_shape[1], int(center.y + half_size))
        
        # Ensure minimum window size
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return None
            
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'center': center
        }
        
    def get_walkable_pixels_in_window(self, pathing_grid: np.ndarray, bounds: Dict) -> List[Tuple[float, float]]:
        """Extract walkable pixel coordinates from window"""
        walkable_pixels = []
        
        for x in range(bounds['x_min'], bounds['x_max']):
            for y in range(bounds['y_min'], bounds['y_max']):
                # SC2 coordinates are (x,y), grids are [x,y] - following the memory about coordinate order
                if pathing_grid[x, y] > 0:  # Walkable tile
                    walkable_pixels.append((float(x), float(y)))
                    
        return walkable_pixels
        
    def apply_pca_to_walkable_area(self, walkable_pixels: List[Tuple[float, float]]) -> Optional[Tuple[Point2, Point2]]:
        """Apply PCA to find principal axis of walkable area"""
        try:
            # Convert to numpy array for PCA
            pixel_array = np.array(walkable_pixels)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca.fit(pixel_array)
            
            # Get principal component (tangent direction)
            principal_component = pca.components_[0]
            tangent_x, tangent_y = float(principal_component[0]), float(principal_component[1])
            
            # Normalize tangent vector
            tangent_length = np.sqrt(tangent_x**2 + tangent_y**2)
            if tangent_length > 0:
                tangent_x /= tangent_length
                tangent_y /= tangent_length
            
            tangent = Point2(tangent_x, tangent_y)
            
            # Calculate normal (perpendicular) - rotate 90 degrees
            normal = Point2(-tangent.y, tangent.x)
            
            return tangent, normal
            
        except Exception as e:
            logger.error(f"PCA analysis failed: {e}")
            return None
            
    def find_narrowest_slice(self, walkable_pixels: List[Tuple[float, float]], 
                           tangent: Point2, normal: Point2, bounds: Dict) -> Optional[ChokeData]:
        """Find the narrowest cross-section by slicing along normal direction"""
        try:
            pixel_array = np.array(walkable_pixels)
            center_point = bounds['center']
            
            min_width = float('inf')
            best_slice_center = None
            
            # Test slices along the principal axis
            num_slices = 20
            slice_range = 10.0  # tiles
            
            for i in range(num_slices):
                # Position along tangent from center
                offset = (i - num_slices / 2) * (slice_range / num_slices)
                slice_center_x = center_point.x + tangent.x * offset
                slice_center_y = center_point.y + tangent.y * offset
                slice_center = Point2(slice_center_x, slice_center_y)
                
                # Measure width at this slice
                width = self.measure_width_at_slice(pixel_array, slice_center, normal)
                
                if width is not None and width < min_width:
                    min_width = width
                    best_slice_center = slice_center
                    
            if best_slice_center is None or min_width == float('inf'):
                return None
                
            return ChokeData(
                center=best_slice_center,
                width=min_width,
                tangent=tangent,
                normal=normal
            )
            
        except Exception as e:
            logger.error(f"Error finding narrowest slice: {e}")
            return None
            
    def measure_width_at_slice(self, pixel_array: np.ndarray, slice_center: Point2, normal: Point2) -> Optional[float]:
        """Measure choke width at specific slice position"""
        try:
            # Project all pixels onto the normal direction relative to slice center
            slice_point = np.array([slice_center.x, slice_center.y])
            normal_vec = np.array([normal.x, normal.y])
            
            # Calculate distance from each pixel to the slice line
            relative_positions = pixel_array - slice_point
            projections = np.dot(relative_positions, normal_vec)
            
            # Find pixels close to this slice (within 1 tile of slice line)
            tangent_vec = np.array([-normal.y, normal.x])  # Perpendicular to normal
            tangent_distances = np.abs(np.dot(relative_positions, tangent_vec))
            
            nearby_pixels = projections[tangent_distances <= 1.0]
            
            if len(nearby_pixels) < 2:
                return None
                
            # Width is the range of projections onto normal
            width = float(np.max(nearby_pixels) - np.min(nearby_pixels))
            return width
            
        except Exception as e:
            logger.error(f"Error measuring width: {e}")
            return None
            
    def snap_to_buildable_tile(self, ideal_pos: Point2, map_data) -> Optional[Point2]:
        """
        Snap ideal position to nearest buildable tile using SC2MapAnalysis.
        
        Uses efficient spatial indexing instead of building our own KD-tree.
        """
        try:
            # Get placement grid from bot
            placement_grid = self.bot.game_info.placement_grid
            
            # Search radius for nearby buildable tiles
            search_radius = 5
            best_pos = None
            min_distance = float('inf')
            
            x_center, y_center = int(ideal_pos.x), int(ideal_pos.y)
            
            # Search in expanding square around ideal position
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    test_x = x_center + dx
                    test_y = y_center + dy
                    
                    # Check bounds
                    if (0 <= test_x < placement_grid.width and 
                        0 <= test_y < placement_grid.height):
                        
                        # Check if buildable (placement_grid uses [y][x] indexing)
                        if placement_grid[test_y][test_x]:
                            test_pos = Point2(float(test_x), float(test_y))
                            distance = ideal_pos.distance_to(test_pos)
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_pos = test_pos
            
            if best_pos is not None:
                # Convert to world coordinates (3x3 building centers)
                world_pos = Point2(float(best_pos.x + 1.5), float(best_pos.y + 1.5))
                return world_pos
            else:
                logger.warning(f"Could not find buildable tile near {ideal_pos}")
                return None
                
        except Exception as e:
            logger.error(f"Error snapping to buildable tile: {e}")
            return None
    
    def determine_building_layout(self, choke_width: float, opponent_race: str) -> List[str]:
        """
        Determine building types based on choke width and opponent race.
        
        Default pattern: Gateway → Cybernetics Core → Gateway (3×3 chain)
        Based on updated spec and wall-off examples provided.
        """
        if choke_width <= 7:
            # Narrow choke: Gateway + Core is sufficient with zealot gap
            return ["GATEWAY", "CYBERNETICSCORE"] 
        elif choke_width >= 12:
            # Wide choke: Gate-Core-Gate + potentially one more building
            buildings = ["GATEWAY", "CYBERNETICSCORE", "GATEWAY"]
            if choke_width >= 16:
                # Very wide: add a fourth building if needed
                buildings.append("FORGE")  # Or could be ROBOTICSFACILITY
            return buildings
        else:
            # Normal choke (8-11 tiles): Standard Gate-Core-Gate pattern
            return ["GATEWAY", "CYBERNETICSCORE", "GATEWAY"]
    
    async def calculate_wall_positions(self, choke_data: ChokeData, buildings: List[str], map_data) -> Optional[WallPositions]:
        """
        Calculate actual wall positions using SC2MapAnalysis spatial tools.
        
        Positions buildings defensively (toward natural) with proper gaps and pylon coverage.
        Based on wall-off examples: probe shows gatekeeper position in 1-tile gap.
        """
        try:
            center = choke_data.center
            
            # Defensive positioning: bias toward natural (opposite to normal direction)
            defensive_offset = choke_data.normal * -2.0  # Move 2 tiles toward natural
            defensive_center = center + defensive_offset
            
            # Calculate building positions along tangent line
            three_by_three_positions = []
            building_spacing = 3.5  # Slightly wider spacing to ensure proper gaps
            
            for i, building in enumerate(buildings):
                # Center buildings around defensive position
                offset = (i - (len(buildings) - 1) / 2) * building_spacing
                building_pos = defensive_center + choke_data.tangent * offset
                
                # Snap to nearest buildable tile using SC2MapAnalysis
                snapped_pos = self.snap_to_buildable_tile(building_pos, map_data)
                if snapped_pos is not None:
                    three_by_three_positions.append([snapped_pos.x, snapped_pos.y])
                else:
                    # Fallback to unsnapped position if snapping fails
                    three_by_three_positions.append([building_pos.x, building_pos.y])
            
            # Calculate pylon positions (behind the wall, toward natural)
            pylon_behind_offset = choke_data.normal * -4.0  # 4 tiles toward natural
            first_pylon_pos = defensive_center + pylon_behind_offset
            
            # Additional pylons if needed (spread slightly)
            pylon_positions = [[first_pylon_pos.x, first_pylon_pos.y]]
            if len(buildings) >= 3:
                # Add second pylon for coverage of wider walls
                second_pylon_pos = first_pylon_pos + choke_data.tangent * 2.0
                pylon_positions.append([second_pylon_pos.x, second_pylon_pos.y])
            
            # Static defence position (behind seam, toward natural)
            static_defence_offset = choke_data.normal * -3.0
            static_defence_pos = defensive_center + static_defence_offset
            static_defence_positions = [[static_defence_pos.x, static_defence_pos.y]]
            
            # Gatekeeper position: 1-tile gap between first two buildings
            if len(three_by_three_positions) >= 2:
                # Gap between first two buildings (typically Gateway and Core)
                building1 = three_by_three_positions[0]
                building2 = three_by_three_positions[1]
                gap_x = (building1[0] + building2[0]) / 2
                gap_y = (building1[1] + building2[1]) / 2
                gatekeeper_positions = [[gap_x, gap_y]]
            else:
                # Single building case: place in front of building
                gatekeeper_positions = [[defensive_center.x, defensive_center.y - 2]]
            
            return WallPositions(
                first_pylon=pylon_positions[:1],  # First pylon only
                pylons=pylon_positions[1:] if len(pylon_positions) > 1 else pylon_positions[:1],
                three_by_threes=three_by_three_positions,
                static_defences=static_defence_positions,
                gate_keeper=gatekeeper_positions
            )
            
        except Exception as e:
            logger.error(f"Error calculating wall positions: {e}")
            return None
    
    async def validate_wall_connectivity(self, wall_positions: WallPositions, opponent_race: str) -> bool:
        """
        Validate that the wall actually blocks enemy units correctly.
        
        Uses flood-fill on dilated grid to ensure proper connectivity.
        """
        try:
            # Get clean pathing grid
            map_data = self.bot.mediator.get_map_data_object
            pathing_grid = map_data.get_pyastar_grid().copy()
            
            # Block building footprints (3x3 buildings)
            for building_pos in wall_positions.three_by_threes:
                self.block_building_footprint(pathing_grid, building_pos, size=3)
            
            # Block pylon footprints (2x2)
            for pylon_pos in wall_positions.first_pylon + wall_positions.pylons:
                self.block_building_footprint(pathing_grid, pylon_pos, size=2)
                
            # If there's a gatekeeper gap, carve it out
            gap_carved = False
            if wall_positions.gate_keeper:
                gap_pos = wall_positions.gate_keeper[0]
                gap_x, gap_y = int(gap_pos[0]), int(gap_pos[1])
                if 0 <= gap_x < pathing_grid.shape[0] and 0 <= gap_y < pathing_grid.shape[1]:
                    pathing_grid[gap_x, gap_y] = 1  # Make walkable
                    gap_carved = True
            
            # Dilate for unit radius (zerglings ~0.5 tiles)
            dilation_kernel = np.ones((2, 2), dtype=bool)  # Small dilation for 0.5 tile radius
            inverted_grid = pathing_grid == 0  # Invert for dilation
            dilated_blocked = binary_dilation(inverted_grid, dilation_kernel)
            dilated_pathing = (~dilated_blocked).astype(int)
            
            # Test connectivity with flood fill
            natural_pos = self.bot.mediator.get_own_nat
            enemy_start = self.bot.enemy_start_locations[0]
            
            connectivity_valid = self.flood_fill_connectivity_test(
                dilated_pathing, enemy_start, natural_pos, gap_carved
            )
            
            if not connectivity_valid:
                logger.warning("Wall failed connectivity validation")
                return False
                
            logger.info("Wall passed connectivity validation")
            return True
            
        except Exception as e:
            logger.error(f"Error in wall connectivity validation: {e}")
            return False
            
    def block_building_footprint(self, grid: np.ndarray, world_pos: List[float], size: int):
        """Block building footprint on grid"""
        try:
            center_x, center_y = world_pos[0], world_pos[1]
            
            # Calculate footprint bounds  
            half_size = size / 2
            min_x = int(center_x - half_size)
            max_x = int(center_x + half_size)
            min_y = int(center_y - half_size)
            max_y = int(center_y + half_size)
            
            # Block tiles in footprint
            for x in range(max(0, min_x), min(grid.shape[0], max_x + 1)):
                for y in range(max(0, min_y), min(grid.shape[1], max_y + 1)):
                    grid[x, y] = 0  # Make unwalkable
                    
        except Exception as e:
            logger.error(f"Error blocking building footprint: {e}")
            
    def flood_fill_connectivity_test(self, grid: np.ndarray, start: Point2, target: Point2, 
                                   gap_should_exist: bool) -> bool:
        """Test if path exists from start to target through wall"""
        try:
            visited = np.zeros_like(grid, dtype=bool)
            stack = [(int(start.x), int(start.y))]
            
            target_x, target_y = int(target.x), int(target.y)
            path_found = False
            
            while stack and not path_found:
                x, y = stack.pop()
                
                if (x < 0 or x >= grid.shape[0] or 
                    y < 0 or y >= grid.shape[1] or
                    visited[x, y] or grid[x, y] == 0):
                    continue
                    
                visited[x, y] = True
                
                # Check if we reached target area
                if abs(x - target_x) <= 3 and abs(y - target_y) <= 3:
                    path_found = True
                    break
                    
                # Add neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((x + dx, y + dy))
            
            # Validation logic based on gap expectation
            if gap_should_exist:
                # Should be able to path through gap
                return path_found
            else:
                # Should NOT be able to path (hard wall)
                return not path_found
                
        except Exception as e:
            logger.error(f"Error in flood fill test: {e}")
            return False
    
    def determine_spawn_position(self) -> str:
        """Determine if we're UpperSpawn or LowerSpawn based on map position"""
        our_start = self.bot.start_location
        map_center = self.bot.game_info.map_center
        
        # Simple approach: compare Y coordinate to map center
        if our_start.y > map_center.y:
            return "UpperSpawn"
        else:
            return "LowerSpawn"
    
    def update_custom_yaml(self, map_name: str, race: str, wall_positions: WallPositions):
        """Update custom building_placements.yml with new map data"""
        
        # Load existing custom data or create new
        if self.custom_yml_path.exists():
            data = self.load_yaml(self.custom_yml_path)
        else:
            data = {"Protoss": {}}
            
        # Ensure structure exists
        if "Protoss" not in data:
            data["Protoss"] = {}
        if map_name not in data["Protoss"]:
            data["Protoss"][map_name] = {}
            
        wall_type = f"Vs{race}NatWall"
        spawn_pos = self.determine_spawn_position()
        
        # Add wall data in ARES format
        data["Protoss"][map_name][wall_type] = {
            "AvailableVsRaces": [race, "Random"],
            spawn_pos: {
                "FirstPylon": wall_positions.first_pylon,
                "Pylons": wall_positions.pylons,
                "ThreeByThrees": wall_positions.three_by_threes,
                "StaticDefences": wall_positions.static_defences,
                "GateKeeper": wall_positions.gate_keeper
            }
        }
        
        # Write back to file
        self.save_yaml(data, self.custom_yml_path)
        
        logger.info(f"Updated building_placements.yml with {map_name} vs {race} wall data")

    async def get_opponent_race_string(self) -> str:
        """Get opponent race as string for wall generation"""
        if self.bot.enemy_race == Race.Zerg:
            return "Zerg"
        elif self.bot.enemy_race == Race.Terran:
            return "Terran"  
        elif self.bot.enemy_race == Race.Protoss:
            return "Protoss"
        else:  # Random or Unknown
            return "Zerg"  # Default to Zerg walls for now
