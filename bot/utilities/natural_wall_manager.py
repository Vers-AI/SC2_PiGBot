"""
Purpose: Generate map-agnostic Protoss natural wall placements dynamically
Key Decisions: Uses SC2MapAnalysis for spatial operations, integrates with ARES via YAML generation
Limitations: Currently only supports Protoss natural walls, requires on_start initialization
"""

import os
import logging
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import traceback
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA

# SC2 imports
from sc2.position import Point2
from sc2.data import Race

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
                
            logger.info(f"Generated wall positions: {wall_positions}")
                
            # Update custom YAML file
            self.update_custom_yaml(map_name, opponent_race, wall_positions)
            self.generated_maps.add(cache_key)
            
            logger.info(f"Successfully generated wall placement for {map_name} vs {opponent_race}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating wall placement for {map_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            
            logger.info(f"Saving YAML data to {file_path}")
            logger.info(f"Data structure: {type(data)}, keys: {list(data.keys()) if data else 'None'}")
            
            with open(file_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Successfully saved YAML file")
            
        except Exception as e:
            logger.error(f"Failed to save YAML to {file_path}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def generate_wall_placement(self, opponent_race: str, 
                                     target_natural: Optional[Point2] = None, 
                                     enemy_start: Optional[Point2] = None) -> Optional[WallPositions]:
        """
        Generate wall placement using choke detection algorithm + SC2MapAnalysis.
        
        Args:
            opponent_race: Race to defend against
            target_natural: Natural expansion to defend (defaults to our natural)
            enemy_start: Enemy starting position (defaults to detected enemy start)
        """
        try:
            # Get SC2MapAnalysis access
            map_data = self.bot.mediator.get_map_data_object
            
            # 1. Find the actual enemy approach path
            # Use provided positions or default to current spawn setup
            enemy_start_pos = enemy_start or self.bot.enemy_start_locations[0]
            natural_pos = target_natural or self.bot.mediator.get_own_nat
            
            # Use ARES pathfinding to get the real path enemies will take
            path = self.bot.mediator.find_raw_path(
                start=enemy_start_pos,
                target=natural_pos,
                grid=map_data.get_pyastar_grid(),
                sensitivity=1  # Default sensitivity for pathfinding (must be integer)
            )
            
            if not path or len(path) < 3:
                logger.warning("Could not find valid path from enemy to natural")
                return None
                
            # DEBUG: Log path details
            logger.info(f"🛣️ Path from {enemy_start_pos} to {natural_pos}:")
            logger.info(f"   Path length: {len(path)} points")
            logger.info(f"   First few points: {path[:3]}")
            logger.info(f"   Last few points: {path[-3:]}")
                
            # 2. Use ARES built-in choke detection instead of custom algorithm
            logger.info(f"🔍 Using ARES get_map_choke_points...")
            
            # Get all chokes on the map from ARES
            try:
                choke_points = self.bot.mediator.get_map_choke_points
                logger.info(f"📍 ARES found {len(choke_points)} choke points on map")
                
                # Find choke closest to natural in direction of enemy spawn
                closest_choke = None
                min_distance = float('inf')
                
                # Calculate direction from natural toward enemy spawn
                enemy_direction = enemy_start_pos - natural_pos
                enemy_direction = enemy_direction / enemy_direction.length  # Normalize
                
                logger.info(f"🧭 Enemy direction from natural: {enemy_direction}")
                
                for choke in choke_points:
                    # Vector from natural to this choke
                    choke_direction = choke - natural_pos
                    
                    # Only consider chokes that are somewhat in the direction of the enemy
                    if choke_direction.length > 0:
                        choke_direction_normalized = choke_direction / choke_direction.length
                        # Dot product: 1 = same direction, -1 = opposite, 0 = perpendicular
                        alignment = (enemy_direction.x * choke_direction_normalized.x + 
                                   enemy_direction.y * choke_direction_normalized.y)
                        
                        distance = natural_pos.distance_to(choke)
                        
                        # Only consider chokes that are roughly toward enemy (alignment > 0.3)
                        # and prioritize closer ones
                        if alignment > 0.3 and distance < min_distance:
                            min_distance = distance
                            closest_choke = choke
                            logger.info(f"   → Better choke: {choke} (distance {distance:.1f}, alignment {alignment:.2f})")
                        
                if closest_choke is None:
                    logger.error("❌ No chokes found by ARES")
                    return None
                    
                logger.info(f"🎯 Selected closest choke: {closest_choke} (distance: {min_distance:.1f})")
                
                # Calculate ACTUAL choke width from available choke points
                choke_width = self.calculate_choke_width(choke_points, closest_choke, natural_pos)
                
                # Create choke data with calculated width
                from types import SimpleNamespace
                choke_data = SimpleNamespace()
                choke_data.center = closest_choke
                choke_data.width = choke_width
                choke_data.tangent = Point2((1, 0))  # Will be refined based on choke geometry
                choke_data.normal = Point2((0, 1))   # Will be refined based on choke geometry
                
                logger.info(f"📐 Calculated choke width: {choke_width:.1f} tiles")
                
            except Exception as e:
                logger.error(f"❌ Error using ARES choke detection: {e}")
                # Fallback to old method
                choke_data = self.find_choke_along_path(path, map_data)
                if choke_data is None:
                    logger.warning("Could not detect choke along enemy path")
                    return None
                
            # 3. Determine building layout based on choke width and race
            choke_width = choke_data.width
            building_width = 3.0  # Each building is 3 tiles wide
            gatekeeper_gap = 1.0  # 1 tile for gatekeeper
            
            # Calculate needed buildings: (choke_width - gatekeeper_gap) / building_width
            needed_buildings = max(2, int((choke_width - gatekeeper_gap) / building_width))
            logger.info(f"📏 Choke width: {choke_width}, calculated needed buildings: {needed_buildings}")
            
            buildings = self.determine_building_layout(choke_width, opponent_race, needed_buildings)
            
            # 4. Calculate actual wall positions using SC2MapAnalysis
            wall_positions = await self.calculate_wall_positions(choke_data, buildings, map_data, natural_pos, choke_points, closest_choke)
            
            if wall_positions is None:
                logger.warning("Could not calculate valid wall positions")
                return None
                
            # DEBUG: Log final wall positions
            logger.info(f"🏗️ Final wall positions:")
            logger.info(f"   First pylon: {wall_positions.first_pylon}")
            logger.info(f"   Pylons: {wall_positions.pylons}")
            logger.info(f"   Buildings: {wall_positions.three_by_threes}")
            logger.info(f"   Gate keeper: {wall_positions.gate_keeper}")
                
            # 5. Validate wall connectivity (will implement later)
            if not await self.validate_wall_connectivity(wall_positions, opponent_race):
                logger.warning("Generated wall failed connectivity validation")
                # TODO: Implement fallback generation
                return None
                
            return wall_positions
            
        except Exception as e:
            logger.error(f"Error in generate_wall_placement: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            
            # Validate pathing grid
            if pathing_grid is None:
                logger.error("Pathing grid is None")
                return None
            if not hasattr(pathing_grid, 'shape') or len(pathing_grid.shape) != 2:
                logger.error(f"Invalid pathing grid shape: {pathing_grid}")
                return None
                
            logger.info(f"Pathing grid shape: {pathing_grid.shape}, dtype: {pathing_grid.dtype}")
            
            # Extract window around the path midpoint
            window_size = 25  # 25x25 tile window
            path_center = path[len(path) // 2]
            # Ensure grid shape is properly accessed
            grid_shape = (int(pathing_grid.shape[0]), int(pathing_grid.shape[1]))
            window_bounds = self.extract_analysis_window(path_center, window_size, grid_shape)
            
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
        
        for x in range(int(bounds['x_min']), int(bounds['x_max'])):
            for y in range(int(bounds['y_min']), int(bounds['y_max'])):
                # SC2 coordinates are (x,y), grids are [x,y] - following the memory about coordinate order
                if x < pathing_grid.shape[0] and y < pathing_grid.shape[1] and pathing_grid[x, y] > 0:  # Walkable tile
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
            
            # Point2 constructor expects a single tuple argument
            tangent = Point2((tangent_x, tangent_y))
            
            # Calculate normal (perpendicular) - rotate 90 degrees
            normal = Point2((-tangent.y, tangent.x))
            
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
                slice_center = Point2((slice_center_x, slice_center_y))
                
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
            
            # Validate placement grid
            if not hasattr(placement_grid, 'data_numpy'):
                logger.error("Placement grid missing data_numpy attribute")
                return None
                
            grid_data = placement_grid.data_numpy
            
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
                    if (0 <= test_x < grid_data.shape[1] and 
                        0 <= test_y < grid_data.shape[0]):
                        
                        # Check if buildable (grid_data uses [y][x] indexing)
                        if grid_data[test_y][test_x]:
                            test_pos = Point2((float(test_x), float(test_y)))
                            distance = ideal_pos.distance_to(test_pos)
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_pos = test_pos
            
            if best_pos is not None:
                # Convert to world coordinates (3x3 building centers)
                world_pos = Point2((float(best_pos.x + 1.5), float(best_pos.y + 1.5)))
                return world_pos
            else:
                logger.warning(f"Could not find buildable tile near {ideal_pos}")
                return None
                
        except Exception as e:
            logger.error(f"Error snapping to buildable tile: {e}")
            return None
    
    def calculate_choke_width(self, choke_points: List[Point2], selected_choke: Point2, natural_pos: Point2) -> float:
        """
        Calculate the actual width of the choke by analyzing nearby choke points.
        
        This finds choke points that form the edges of the same choke opening.
        """
        try:
            # Find other choke points that are close to the selected choke (part of same opening)
            choke_cluster = [selected_choke]
            search_radius = 8.0  # Look for choke points within 8 tiles
            
            for point in choke_points:
                distance = selected_choke.distance_to(point)
                if 1.0 < distance <= search_radius:  # Close but not the same point
                    choke_cluster.append(point)
            
            if len(choke_cluster) < 2:
                logger.warning(f"Only found {len(choke_cluster)} choke points, using default width")
                return 6.0  # Default fallback width
            
            # Calculate the spread of choke points (approximate choke width)
            min_x = min(p.x for p in choke_cluster)
            max_x = max(p.x for p in choke_cluster)
            min_y = min(p.y for p in choke_cluster)
            max_y = max(p.y for p in choke_cluster)
            
            # Use the larger dimension as choke width
            width_x = max_x - min_x
            width_y = max_y - min_y
            choke_width = max(width_x, width_y)
            
            # Clamp to reasonable range
            choke_width = max(4.0, min(choke_width, 15.0))
            
            logger.info(f"📐 Choke analysis: {len(choke_cluster)} points, X spread: {width_x:.1f}, Y spread: {width_y:.1f}")
            logger.info(f"📐 Final choke width: {choke_width:.1f} tiles")
            
            return choke_width
            
        except Exception as e:
            logger.error(f"Error calculating choke width: {e}")
            return 6.0  # Safe fallback
    
    def calculate_choke_depth(self, choke_points: List[Point2], selected_choke: Point2, normal_direction: Point2) -> float:
        """
        Calculate the depth (vertical spread) of the choke opening.
        
        This projects choke points onto the normal direction to find how 'thick' the choke is.
        """
        try:
            # Find choke points that are part of the same opening
            choke_cluster = [selected_choke]
            search_radius = 8.0
            
            for point in choke_points:
                distance = selected_choke.distance_to(point)
                if 1.0 < distance <= search_radius:
                    choke_cluster.append(point)
            
            if len(choke_cluster) < 2:
                return 3.0  # Default shallow depth
            
            # Project all choke points onto the normal direction to find depth spread
            projections = []
            for point in choke_cluster:
                # Vector from selected choke to this point
                offset = point - selected_choke
                # Project onto normal direction (dot product)
                projection = offset.x * normal_direction.x + offset.y * normal_direction.y
                projections.append(projection)
            
            # Calculate the spread of projections (this is the choke depth)
            min_proj = min(projections)
            max_proj = max(projections)
            depth = max_proj - min_proj
            
            # Clamp to reasonable range
            depth = max(1.0, min(depth, 6.0))
            
            logger.info(f"📐 Choke depth analysis: {len(choke_cluster)} points, projection spread: {depth:.1f}")
            
            return depth
            
        except Exception as e:
            logger.error(f"Error calculating choke depth: {e}")
            return 3.0  # Safe fallback

    def try_place_building_at_position(self, pos: Point2, building_type: str, existing_positions: List[List[float]], round_coord_func) -> bool:
        """
        Try to place a building at the given position.
        Returns True if successful, False if position is invalid.
        """
        try:
            # Check if position is buildable
            if not self.bot.mediator.can_place_structure(position=pos, structure_type=self.get_unit_type(building_type)):
                return False
            
            # Check for overlaps with existing buildings
            for existing_pos in existing_positions:
                existing_point = Point2((existing_pos[0], existing_pos[1]))
                distance = pos.distance_to(existing_point)
                if distance < 3.0:  # 3x3 buildings need at least 3 tile separation
                    return False
            
            # Position is valid - add it to the list with proper snapping
            existing_positions.append([round_coord_func(pos.x), round_coord_func(pos.y)])
            return True
            
        except Exception as e:
            logger.error(f"Error checking building position {pos}: {e}")
            return False

    def get_unit_type(self, building_type: str):
        """Convert building type string to UnitTypeId"""
        from sc2.ids.unit_typeid import UnitTypeId
        
        building_map = {
            "GATEWAY": UnitTypeId.GATEWAY,
            "CYBERNETICSCORE": UnitTypeId.CYBERNETICSCORE,
            "FORGE": UnitTypeId.FORGE,
            "ROBOTICSFACILITY": UnitTypeId.ROBOTICSFACILITY,
            "PYLON": UnitTypeId.PYLON
        }
        return building_map.get(building_type, UnitTypeId.GATEWAY)

    def find_non_overlapping_position(self, ideal_pos: Point2, building_type: str, 
                                     existing_positions: List[List[float]], search_radius: int = 3) -> Optional[Point2]:
        """
        Find a valid building position that doesn't overlap with existing buildings.
        
        Args:
            ideal_pos: Preferred position for the building
            building_type: Type of building (e.g., "GATEWAY", "CYBERNETICSCORE")
            existing_positions: List of already placed building positions [[x, y], ...]
            search_radius: How far to search for a valid position
            
        Returns:
            Valid Point2 position or None if no valid position found
        """
        try:
            # Map building types to UnitTypeId
            from sc2.ids.unit_typeid import UnitTypeId
            
            building_map = {
                "GATEWAY": UnitTypeId.GATEWAY,
                "CYBERNETICSCORE": UnitTypeId.CYBERNETICSCORE,
                "FORGE": UnitTypeId.FORGE,
                "ROBOTICSFACILITY": UnitTypeId.ROBOTICSFACILITY,
                "PYLON": UnitTypeId.PYLON
            }
            
            unit_type = building_map.get(building_type, UnitTypeId.GATEWAY)
            building_size = 3.0  # Most buildings are 3x3
            min_distance = building_size  # Buildings can be adjacent (touching edge-to-edge)
            
            # Search in expanding circle around ideal position
            for radius in range(search_radius + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Only check positions on the current radius circle
                        if abs(dx) != radius and abs(dy) != radius and radius > 0:
                            continue
                            
                        test_pos = Point2((ideal_pos.x + dx, ideal_pos.y + dy))
                        
                        # Check if position is placeable
                        if not self.bot.mediator.can_place_structure(
                            position=test_pos,
                            structure_type=unit_type
                        ):
                            continue
                            
                        # Check for overlaps with existing buildings
                        overlap_found = False
                        for existing_pos in existing_positions:
                            existing_point = Point2((existing_pos[0], existing_pos[1]))
                            distance = test_pos.distance_to(existing_point)
                            
                            if distance < min_distance:
                                overlap_found = True
                                break
                                
                        if not overlap_found:
                            return test_pos
                            
            logger.warning(f"No non-overlapping position found for {building_type} near {ideal_pos}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding non-overlapping building position: {e}")
            return None

    def find_valid_building_position(self, ideal_pos: Point2, building_type: str, search_radius: int = 5) -> Optional[Point2]:
        """
        Find a valid building position near the ideal location using ARES placement validation.
        
        Args:
            ideal_pos: Preferred position for the building
            building_type: Type of building (e.g., "GATEWAY", "CYBERNETICSCORE")
            search_radius: How far to search for a valid position
            
        Returns:
            Valid Point2 position or None if no valid position found
        """
        try:
            # Map building types to UnitTypeId (need to import this)
            from sc2.ids.unit_typeid import UnitTypeId
            
            building_map = {
                "GATEWAY": UnitTypeId.GATEWAY,
                "CYBERNETICSCORE": UnitTypeId.CYBERNETICSCORE,
                "FORGE": UnitTypeId.FORGE,
                "ROBOTICSFACILITY": UnitTypeId.ROBOTICSFACILITY,
                "PYLON": UnitTypeId.PYLON
            }
            
            unit_type = building_map.get(building_type, UnitTypeId.GATEWAY)
            
            # Search in expanding circle around ideal position
            for radius in range(search_radius + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Only check positions on the current radius circle
                        if abs(dx) != radius and abs(dy) != radius and radius > 0:
                            continue
                            
                        test_pos = Point2((ideal_pos.x + dx, ideal_pos.y + dy))
                        
                        # Use ARES placement validation
                        if self.bot.mediator.can_place_structure(
                            position=test_pos,
                            structure_type=unit_type
                        ):
                            return test_pos
                            
            logger.warning(f"No valid position found for {building_type} near {ideal_pos}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding valid building position: {e}")
            return None
    
    def determine_building_layout(self, choke_width: float, opponent_race: str, needed_buildings: Optional[int] = None) -> List[str]:
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
    
    async def calculate_wall_positions(self, choke_data: ChokeData, buildings: List[str], map_data, natural_pos: Point2, choke_points: Optional[List[Point2]] = None, selected_choke: Optional[Point2] = None) -> Optional[WallPositions]:
        """
        Calculate actual wall positions using SC2MapAnalysis spatial tools.
        
        Positions buildings defensively (toward natural) with proper gaps and pylon coverage.
        Based on wall-off examples: probe shows gatekeeper position in 1-tile gap.
        """
        try:
            center = choke_data.center
            
            # Helper function to round coordinates based on building type
            def round_coord_for_building(val: float, building_type: str) -> float:
                """Round coordinate based on SC2 building snap rules"""
                if building_type == "PYLON":  # 2x2 buildings snap to .0
                    return float(round(val))
                else:  # 3x3, 5x5, and GATEKEEPER all snap to .5
                    return float(round(val - 0.5) + 0.5)
            
            def round_coord(val: float) -> float:
                """Default rounding for 3x3 buildings (.5 snap)"""
                return float(round(val - 0.5) + 0.5)
            
            # Position buildings TO COVER THE CHOKE OPENING, not just near it
            # Buildings must span across the actual choke passage
            natural_to_choke = center - natural_pos
            
            if natural_to_choke.length > 0:
                natural_to_choke_normalized = natural_to_choke / natural_to_choke.length
                
                # Position wall AT the choke to actually block it, but slightly toward natural for defense
                wall_distance_from_natural = natural_to_choke.length * 0.85  # 85% of way - closer to choke
                defensive_center = natural_pos + natural_to_choke_normalized * wall_distance_from_natural
                
                logger.info(f"🛡️ Choke-blocking positioning:")
                logger.info(f"   Natural pos: {natural_pos}")
                logger.info(f"   Choke center: {center}")  
                logger.info(f"   Natural→Choke direction: {natural_to_choke_normalized}")
                logger.info(f"   Wall distance from natural: {wall_distance_from_natural} (85% to choke)")
                logger.info(f"   Wall center (closer to choke): {defensive_center}")
            else:
                defensive_center = center
                logger.warning("⚠️ Could not determine natural-to-choke direction, using choke center")
            
            # Calculate building positions to SPAN ACROSS the choke opening
            # Buildings must be positioned along the choke tangent to block the passage
            choke_width = choke_data.width
            building_width = 3.0  # Each building covers 3 tiles
            
            logger.info(f"📐 Choke geometry: width={choke_width}, tangent={choke_data.tangent}")
            
            # Position buildings to span the full choke width, leaving 1-tile gatekeeper gap
            three_by_three_positions = []
            gatekeeper_gap_pos = None
            
            if len(buildings) == 2:
                # 2 buildings: space them exactly 3.0 tiles apart (edge-to-edge touching)
                building_spacing = 3.0  # 3x3 buildings touching edge-to-edge
                building_positions = [
                    defensive_center - choke_data.tangent * building_spacing,  # Left building
                    defensive_center + choke_data.tangent * building_spacing   # Right building  
                ]
                gatekeeper_gap_pos = defensive_center  # Gap at center
                
            elif len(buildings) == 3:
                # 3 buildings: use 2D positioning to fill choke shape, not just span width
                min_spacing = 3.0  # Minimum for touching buildings
                max_spacing = 4.0  # Maximum to prevent huge gaps
                
                # Calculate ideal spacing but clamp it
                coverage_width = max(6.0, choke_width - 1.0)  # Ensure minimum coverage
                ideal_spacing = coverage_width / 2
                building_spacing = max(min_spacing, min(ideal_spacing, max_spacing))
                
                # Base positions along tangent (horizontal spread)
                base_positions = [
                    defensive_center - choke_data.tangent * building_spacing,      # Left
                    defensive_center,                                              # Center
                    defensive_center + choke_data.tangent * building_spacing       # Right
                ]
                
                # Calculate vertical offsets based on actual choke geometry
                if choke_points and selected_choke:
                    vertical_depth = self.calculate_choke_depth(choke_points, selected_choke, choke_data.normal)
                else:
                    vertical_depth = 3.0  # Default depth if choke data not available
                
                # Distribute buildings across the vertical depth to fill gaps
                if vertical_depth > 2.0:  # Only offset if choke has significant depth
                    max_offset = min(vertical_depth / 2, 2.0)  # Cap at 2 tiles max
                    vertical_offsets = [
                        -max_offset * 0.5,  # Left building: toward enemy side
                        0.0,                # Center building: stays on line
                        max_offset * 0.5    # Right building: toward natural side
                    ]
                    logger.info(f"📐 Choke depth: {vertical_depth:.1f}, using offsets: {vertical_offsets}")
                else:
                    # Shallow choke - keep buildings aligned
                    vertical_offsets = [0.0, 0.0, 0.0]
                    logger.info(f"📐 Shallow choke (depth: {vertical_depth:.1f}), no vertical offsets needed")
                
                building_positions = []
                for i, (base_pos, offset) in enumerate(zip(base_positions, vertical_offsets)):
                    # Apply vertical offset along the normal direction
                    final_pos = base_pos + choke_data.normal * offset
                    building_positions.append(final_pos)
                    logger.info(f"🏗️ Building {i+1}: base {base_pos} + vertical offset {offset} = {final_pos}")
                
                logger.info(f"📏 Choke width: {choke_width}, coverage needed: {coverage_width}")
                logger.info(f"📏 Spacing: ideal={ideal_spacing:.1f}, clamped={building_spacing:.1f}")
                logger.info(f"🔀 Using 2D formation to fill vertical gaps")
                
                # Place gatekeeper in the 1-tile gap (between left and center buildings)
                gatekeeper_gap_pos = defensive_center - choke_data.tangent * (building_spacing / 2)
                
            else:
                # Fallback: distribute evenly across choke
                if len(buildings) > 0:
                    total_span = min(choke_width * 0.8, len(buildings) * building_width)  # Don't exceed choke
                    spacing = total_span / len(buildings) if len(buildings) > 1 else 0
                    building_positions = []
                    for i in range(len(buildings)):
                        offset = (i - (len(buildings) - 1) / 2) * spacing
                        building_positions.append(defensive_center + choke_data.tangent * offset)
                    gatekeeper_gap_pos = defensive_center
                else:
                    building_positions = []
                    
            logger.info(f"🏗️ Building span calculation: {len(buildings)} buildings across {choke_width} width")
            
            # Validate and place each building with fallback search
            for i, (building, ideal_pos) in enumerate(zip(buildings, building_positions)):
                building_placed = False
                
                # Try the ideal position first
                building_round_func = lambda x: round_coord_for_building(x, building)
                if self.try_place_building_at_position(ideal_pos, building, three_by_three_positions, building_round_func):
                    building_placed = True
                    logger.info(f"✅ {building} placed at ideal position {ideal_pos}")
                else:
                    # Search nearby for a valid position
                    logger.info(f"🔍 Ideal position {ideal_pos} not buildable, searching nearby...")
                    
                    # Try positions in expanding circle around ideal position
                    for search_radius in [1, 2, 3]:
                        found_position = False
                        for dx in range(-search_radius, search_radius + 1):
                            for dy in range(-search_radius, search_radius + 1):
                                if dx == 0 and dy == 0:  # Skip center (already tried)
                                    continue
                                    
                                candidate_pos = Point2((ideal_pos.x + dx, ideal_pos.y + dy))
                                
                                if self.try_place_building_at_position(candidate_pos, building, three_by_three_positions, building_round_func):
                                    building_placed = True
                                    found_position = True
                                    logger.info(f"✅ {building} placed at fallback position {candidate_pos} (offset: {dx}, {dy})")
                                    break
                            if found_position:
                                break
                        if found_position:
                            break
                
                if not building_placed:
                    logger.warning(f"❌ Could not find buildable position for {building} near {ideal_pos}")
            
            # Initialize position arrays
            first_pylon_positions = []
            pylon_positions = []
            static_defence_positions = []
            
            # Calculate pylon positions (behind the wall, toward natural)
            # IMPORTANT: Avoid the natural expansion's 5x5 footprint
            if natural_to_choke.length > 0:
                # Recalculate direction for clarity (same as above)
                natural_to_choke_direction = natural_to_choke / natural_to_choke.length
                toward_natural = natural_to_choke_direction * -1.0  # Flip to go toward natural
                
                # Try different distances to avoid natural's 5x5 area
                for pylon_distance in [4.0, 5.0, 6.0, 7.0]:  # Try increasing distances
                    candidate_pylon_pos = defensive_center + toward_natural * pylon_distance
                    
                    # Check if pylon would conflict with natural's 5x5 footprint
                    distance_to_natural = candidate_pylon_pos.distance_to(natural_pos)
                    natural_radius = 3.0  # 5x5 building has ~2.5 radius, use 3.0 for safety
                    
                    if distance_to_natural > natural_radius:
                        valid_first_pylon = candidate_pylon_pos
                        first_pylon_positions.append([round_coord_for_building(valid_first_pylon.x, "PYLON"), round_coord_for_building(valid_first_pylon.y, "PYLON")])
                        logger.info(f"✅ First pylon positioned at safe distance {pylon_distance} from natural")
                        
                        # Add to regular pylon array as well (first pylon can serve as additional power)
                        pylon_positions = [[round_coord_for_building(valid_first_pylon.x, "PYLON"), round_coord_for_building(valid_first_pylon.y, "PYLON")]]
                        
                        break
                    else:
                        # If all distances conflict, try perpendicular placement
                        perpendicular = Point2((-natural_to_choke_direction.y, natural_to_choke_direction.x))
                        first_pylon_pos = defensive_center + perpendicular * 4.0
                        logger.warning(f"⚠️ Pylon placed perpendicular to avoid natural: {first_pylon_pos}")
            else:
                # Fallback: use normal vector
                pylon_behind_offset = choke_data.normal * -4.0  
                first_pylon_pos = defensive_center + pylon_behind_offset
            
            # Validate that we have some buildings placed
            if not three_by_three_positions:
                logger.error("❌ No valid building positions found - cannot create wall")
                return None
            
            # Static defence position (behind seam, toward natural)
            static_defence_offset = choke_data.normal * -3.0
            static_defence_pos = defensive_center + static_defence_offset
            static_defence_positions = [[round_coord(static_defence_pos.x), round_coord(static_defence_pos.y)]]
            
            # Calculate gatekeeper position (probe gap)
            if gatekeeper_gap_pos:
                logger.info(f"🚪 Gatekeeper positioned at intentional gap: {gatekeeper_gap_pos}")
                gate_keeper_positions = [[round_coord_for_building(gatekeeper_gap_pos.x, "GATEKEEPER"), round_coord_for_building(gatekeeper_gap_pos.y, "GATEKEEPER")]]
                logger.info(f"🚪 Gatekeeper positioned at intentional gap: {gatekeeper_gap_pos}")
            else:
                # Fallback: place between first two buildings if available
                if len(three_by_three_positions) >= 2:
                    building1 = three_by_three_positions[0]
                    building2 = three_by_three_positions[1]
                    gap_x = (building1[0] + building2[0]) / 2
                    gap_y = (building1[1] + building2[1]) / 2
                    gate_keeper_positions = [[round_coord(gap_x), round_coord(gap_y)]]
                else:
                    # Single building case: place in front of building
                    gate_keeper_positions = [[round_coord(defensive_center.x), round_coord(defensive_center.y - 2)]]
            
            return WallPositions(
                first_pylon=pylon_positions[:1],  # First pylon only
                pylons=pylon_positions[1:] if len(pylon_positions) > 1 else pylon_positions[:1],
                three_by_threes=three_by_three_positions,
                static_defences=static_defence_positions,
                gate_keeper=gate_keeper_positions
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
                gap_x, gap_y = int(float(gap_pos[0])), int(float(gap_pos[1]))
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
            center_x, center_y = float(world_pos[0]), float(world_pos[1])
            
            # Calculate footprint bounds  
            half_size = size / 2.0
            min_x = int(center_x - half_size)
            max_x = int(center_x + half_size)
            min_y = int(center_y - half_size)
            max_y = int(center_y + half_size)
            
            # Block tiles in footprint
            for x in range(max(0, min_x), min(grid.shape[0], max_x + 1)):
                for y in range(max(0, min_y), min(grid.shape[1], max_y + 1)):
                    if x < grid.shape[0] and y < grid.shape[1]:
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
