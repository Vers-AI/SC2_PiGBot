"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from sc2.ids.ability_id import AbilityId


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.behaviors.combat.individual import PathUnitToTarget
from ares.managers.manager_mediator import ManagerMediator
import numpy as np
from sc2.position import Point2


if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, mediator: ManagerMediator, bot: 'AresBot'):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = 21.4
        self.nova_duration = 2.1
        self.best_target_pos = None
        self.frames_left = 48  # Example starting frame count
        self.distance_left = 0.0
        self.unit = None  # References the Purification Nova ability which is a unit not a spell
        self.mediator = mediator
        self.bot = bot  # Store bot instance
        
        # Do NOT store any references to grids here, get them fresh when needed
        self.best_target_influence = 0
        self.should_debug_visuals = False  # Set to True for visual debugging
        
        # Print initialization message
        print("DEBUG: UseDisruptorNova initialized")
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units: List['Unit'], friendly_units: List['Unit'], exclusion_mask=None) -> Optional[Point2]:
        """Select the best target position for the nova.
        Uses a hybrid approach: grid influence values combined with enemy unit positioning.
        Implemented with optimized vectorized operations for performance.
        
        Args:
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            exclusion_mask: Optional boolean mask where True indicates areas to avoid (already targeted by other Novas)
        
        Returns:
            The Point2 target position or None if no valid target found
        """
        if not enemy_units:
            print("DEBUG select_best_target: No enemy units to target")
            return None

        try:
            # Get the tactical grid which contains influence values
            grid = self.get_grid()
            if grid is None:
                print("DEBUG select_best_target: Warning - tactical grid is None")
                # Fall back to position-based targeting without grid influence
                return self._select_target_position_based(enemy_units, friendly_units, exclusion_mask)
            else:
                # Replace infinite values with large finite values to avoid numerical issues
                grid = np.where(np.isinf(grid), 10000.0, grid)
                print(f"DEBUG select_best_target: Grid info - shape={grid.shape}, min={np.min(grid)}, max={np.max(grid)}")
            
            # Apply the exclusion mask if provided
            if exclusion_mask is not None:
                try:
                    # Check shape compatibility
                    if exclusion_mask.shape != grid.shape:
                        print(f"DEBUG select_best_target: Warning - exclusion mask shape {exclusion_mask.shape} doesn't match grid shape {grid.shape}")
                        # Fall back to position-based targeting
                        return self._select_target_position_based(enemy_units, friendly_units, exclusion_mask)
                    
                    # Create a working copy of the grid with exclusions applied
                    working_grid = grid.copy()
                    # Set excluded areas to -np.inf so they won't be selected
                    working_grid[exclusion_mask] = -np.inf
                    print(f"DEBUG select_best_target: Applied exclusion mask, valid grid area: {np.sum(~exclusion_mask)} cells")
                except Exception as e:
                    print(f"DEBUG ERROR applying exclusion mask: {e}")
                    # Fall back to position-based targeting
                    return self._select_target_position_based(enemy_units, friendly_units, exclusion_mask)
            else:
                # No exclusion mask, use the grid as is
                working_grid = grid.copy()
            
            # Create additional masks for friendly/enemy areas to enhance the grid
            try:
                # Create enemy boost mask - areas with enemies get boosted influence
                enemy_boost_mask = np.zeros_like(working_grid, dtype=bool)
                enemy_positions = np.zeros_like(working_grid, dtype=bool)
                nova_radius = 1.5  # Nova blast radius
                
                # Mark enemy positions in the grid
                for enemy in enemy_units:
                    # Convert to grid coordinates
                    enemy_y = min(max(int(enemy.position.y), 0), working_grid.shape[0] - 1)
                    enemy_x = min(max(int(enemy.position.x), 0), working_grid.shape[1] - 1)
                    enemy_positions[enemy_y, enemy_x] = True
                
                # Create boost areas around enemies using distance transform
                if np.any(enemy_positions):
                    # Create coordinate grids
                    y_indices, x_indices = np.indices(working_grid.shape)
                    
                    # For each enemy, boost a circular area around it
                    for enemy in enemy_units:
                        enemy_y = min(max(int(enemy.position.y), 0), working_grid.shape[0] - 1)
                        enemy_x = min(max(int(enemy.position.x), 0), working_grid.shape[1] - 1)
                        
                        # Calculate distance to this enemy from all points (vectorized)
                        distances = np.sqrt(((y_indices - enemy_y) ** 2) + ((x_indices - enemy_x) ** 2))
                        
                        # Points within nova_radius of enemy get boosted
                        enemy_boost_mask = np.logical_or(enemy_boost_mask, distances <= nova_radius)
                    
                    # Apply enemy boost to grid - increase influence in areas with enemies
                    # This will make the grid influence higher in areas with clustered enemies
                    working_grid[enemy_boost_mask] += 200  # Significant boost to hit areas with enemies
                    print(f"DEBUG select_best_target: Boosted {np.sum(enemy_boost_mask)} cells near enemies")
                
                # Create friendly penalty map instead of complete avoidance
                friendly_penalty_map = np.zeros_like(working_grid)
                friendly_positions = []
                
                # Track friendly positions for penalty calculation
                for friendly in friendly_units:
                    friendly_y = min(max(int(friendly.position.y), 0), working_grid.shape[0] - 1)
                    friendly_x = min(max(int(friendly.position.x), 0), working_grid.shape[1] - 1)
                    friendly_positions.append((friendly_y, friendly_x))
                
                # Calculate penalty for each friendly unit - more penalty for more friendlies
                if friendly_positions:
                    for friendly_y, friendly_x in friendly_positions:
                        # Create coordinate grids for distance calculation
                        y_indices, x_indices = np.indices(working_grid.shape)
                        
                        # Calculate distance to this friendly from all points
                        distances = np.sqrt(((y_indices - friendly_y) ** 2) + ((x_indices - friendly_x) ** 2))
                        
                        # Apply penalty to areas that would hit this friendly unit
                        # Note: 250 penalty per friendly is high but not -np.inf, allowing for cost-benefit analysis
                        friendly_penalty_map = np.where(distances <= nova_radius, 
                                                      friendly_penalty_map + 250, 
                                                      friendly_penalty_map)
                    
                    # Apply the penalty to the working grid
                    working_grid -= friendly_penalty_map
                    print(f"DEBUG select_best_target: Applied friendly penalties to {np.sum(friendly_penalty_map > 0)} cells")
                
                # Explicitly mask out the origin (0,0) to ensure we don't choose it by default
                # This prevents false targeting when no real target is available
                original_origin_value = working_grid[0, 0]
                working_grid[0, 0] = -np.inf
                print(f"DEBUG select_best_target: Masked out origin (0,0) with value {original_origin_value}")
                
                # Find best position
                best_y, best_x = np.unravel_index(np.argmax(working_grid), working_grid.shape)
                best_influence = working_grid[best_y, best_x]
                
                # Check if we have a valid target with sufficient influence
                # Requires both a reasonable influence value and not being at the origin
                if best_influence > 200 and not (best_y == 0 and best_x == 0):
                    best_pos = Point2((float(best_x), float(best_y)))
                    print(f"DEBUG select_best_target: Grid-based approach found target at {best_pos} with influence {best_influence}")
                    
                    # Store for future comparisons
                    self.best_target_pos = best_pos
                    self.best_target_influence = best_influence
                    
                    # Debug visualization if enabled
                    if self.should_debug_visuals:
                        self._visualize_grid(working_grid)
                    
                    return best_pos
                else:
                    print("DEBUG select_best_target: Grid-based approach found no valid target, falling back to position-based")
                    # Fall back to position-based approach
                    fallback_pos = self._select_target_position_based(enemy_units, friendly_units, exclusion_mask)
                    if fallback_pos is not None:
                        print(f"DEBUG select_best_target: Position-based fallback found target at {fallback_pos}")
                    else:
                        print("DEBUG select_best_target: Position-based fallback also failed to find a target")
                    return fallback_pos
            
            except Exception as e:
                print(f"DEBUG ERROR in grid-based targeting: {e}")
                # Fall back to position-based targeting on error
                fallback_pos = self._select_target_position_based(enemy_units, friendly_units, exclusion_mask)
                if fallback_pos is not None:
                    print(f"DEBUG select_best_target: Position-based fallback found target at {fallback_pos}")
                else:
                    print("DEBUG select_best_target: Position-based fallback also failed to find a target")
                return fallback_pos
            
        except Exception as e:
            print(f"DEBUG Error in select_best_target: {e}")
            return None
            
    def _select_target_position_based(self, enemy_units, friendly_units, exclusion_mask=None):
        """Legacy position-based targeting as a fallback when grid-based targeting fails.
        
        Args:
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            exclusion_mask: Optional boolean mask for exclusion zones
            
        Returns:
            Point2 target position or None
        """
        try:
            print(f"DEBUG: Using fallback position-based targeting with {len(enemy_units)} enemies")
            
            # Get grid for boundary information, but we won't use its influence values
            grid = self.get_grid()
            
            # Set up boundaries for candidate positions
            if grid is not None:
                grid_height, grid_width = grid.shape
                x_min, y_min = 0, 0 
                x_max, y_max = grid_width, grid_height
            else:
                # Without a grid, use estimated map boundaries
                x_min, y_min = 0, 0
                x_max, y_max = 200, 200
            
            # Generate candidate positions
            candidate_positions = []
            scores = []
            resolution = 8  # Sampling resolution

            # Include enemy positions and their surroundings
            for enemy in enemy_units:
                pos = enemy.position
                candidate_positions.append(pos)
                
                # Add surrounding positions for better coverage
                for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                    candidate_positions.append(Point2((pos.x + dx, pos.y + dy)))
                
            # Add a systematic grid of positions 
            for x in range(x_min, x_max, resolution):
                for y in range(y_min, y_max, resolution):
                    candidate_positions.append(Point2((x, y)))
            
            # Score each position based on enemy clustering
            nova_radius = 1.5
            for pos in candidate_positions:
                # Skip excluded positions
                if exclusion_mask is not None and grid is not None:
                    try:
                        grid_y = min(max(int(pos.y), 0), grid.shape[0] - 1)
                        grid_x = min(max(int(pos.x), 0), grid.shape[1] - 1)
                        
                        if exclusion_mask[grid_y, grid_x]:
                            continue
                    except Exception as e:
                        print(f"DEBUG ERROR checking exclusion: {e}")
                
                # Count enemies and friendlies hit
                enemies_hit = sum(1 for enemy in enemy_units if pos.distance_to(enemy.position) <= nova_radius)
                friendly_hit = sum(1 for friendly in friendly_units if pos.distance_to(friendly.position) <= nova_radius)
                
                # Use cost-benefit analysis for target selection
                if enemies_hit > 0:
                    # Calculate a score that balances enemy hits vs friendly hits
                    # Each enemy is worth 150 points, each friendly costs 200 points
                    # More lenient weights to improve chances of finding a valid target
                    score = (enemies_hit * 150) - (friendly_hit * 200)
                    
                    # Record the score for all positions for debugging
                    print(f"DEBUG: Position {pos} scores {score} (hits {enemies_hit} enemies, {friendly_hit} friendlies)")
                    
                    # Be more lenient with scoring - allow slight friendly fire if it hits enough enemies
                    if score > -100:  # Even slightly negative scores are acceptable in a pinch
                        scores.append((pos, score))
            
            # Sort by score (highest first)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores:
                best_pos, best_score = scores[0]
                print(f"DEBUG: Position-based targeting selected {best_pos} with score {best_score}")
                
                # Store values for comparisons in other methods
                self.best_target_pos = best_pos
                self.best_target_score = best_score
                
                return best_pos
            else:
                print("DEBUG: Position-based targeting found no valid targets")
                
                # Last resort: if there are any enemies, target the closest enemy
                if enemy_units:
                    closest_enemy = min(enemy_units, key=lambda enemy: self.unit.position.distance_to(enemy.position))
                    target_pos = closest_enemy.position
                    print(f"DEBUG: Last resort - targeting closest enemy at {target_pos}")
                    
                    # Store this last resort target for reference
                    self.best_target_pos = target_pos
                    self.best_target_score = -1  # Negative to indicate it was last resort
                    
                    return target_pos
                # Only reached if no valid targets and no enemies - return None
                print("DEBUG: No valid position-based targets found")
                return None
                
        except Exception as e:
            print(f"DEBUG ERROR in position-based targeting: {e}")
            return None

    def update_target_position(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager):
        """
        Check if a better target has become available within the Nova's remaining travel range.
        Uses vectorized operations for performance and directly leverages grid influence values.
        
        Args:
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            nova_manager: The NovaManager instance
            
        Returns:
            bool: True if the target was updated, False otherwise
        """
        if not self.best_target_pos or not self.unit:
            return False
            
        try:
            # Check if there's enough time to change course
            if self.frames_left < 5:  
                return False
                
            # Get the current position of the Nova
            current_position = self.unit.position
            
            # Calculate the maximum distance the Nova can still travel
            max_travel_distance = nova_manager.nova_speed * (self.frames_left / 22.4)
            
            # Get the tactical grid and create a copy we can modify
            grid = self.get_grid()
            if grid is None:
                print("ERROR: Tactical grid is None")
                return False
            
            # Create coordinate grids and distance mask using vectorized operations
            try:
                # Create coordinate meshgrids for the entire tactical map
                y_indices, x_indices = np.indices(grid.shape)
                
                # Calculate distances from current position to all points vectorized
                # Note: This assumes grid coordinates map directly to game coordinates
                distances = np.sqrt(((x_indices - current_position.x) ** 2) + 
                                  ((y_indices - current_position.y) ** 2))
                
                # Create mask for areas out of reach (True = excluded, False = available)
                out_of_reach_mask = distances > max_travel_distance
                
                print(f"DEBUG: Vectorized distance mask created with {np.sum(~out_of_reach_mask)} positions in range")
            except Exception as e:
                print(f"DEBUG ERROR creating vectorized distance mask: {e}")
                return False
            
            # Temporarily unregister our current target to avoid its exclusion zone
            current_target_temporarily_unregistered = False
            if self.best_target_pos:
                try:
                    nova_manager.unregister_nova_target(self.best_target_pos)
                    current_target_temporarily_unregistered = True
                    print(f"DEBUG: Temporarily unregistered target at {self.best_target_pos} for search")
                except Exception as e:
                    print(f"DEBUG ERROR unregistering target for search: {e}")
            
            # Get the exclusion mask from the nova manager
            try:
                exclusion_mask = nova_manager.get_exclusion_mask(grid)
                combined_mask = np.logical_or(exclusion_mask, out_of_reach_mask)
            except Exception as e:
                print(f"DEBUG ERROR getting exclusion mask: {e}")
                combined_mask = out_of_reach_mask
            
            # Use the grid influence values directly to find the best target
            try:
                # Create mask of valid positions (inverse of combined_mask)
                valid_positions_mask = ~combined_mask
                
                if np.any(valid_positions_mask):
                    # Apply the mask to the grid - set invalid positions to negative infinity
                    # so they won't be selected as maximum
                    masked_grid = np.where(valid_positions_mask, grid, -np.inf)
                    
                    # Find position with maximum influence in the valid area
                    max_y, max_x = np.unravel_index(np.argmax(masked_grid), grid.shape)
                    new_target = Point2((float(max_x), float(max_y)))
                    new_influence = masked_grid[max_y, max_x]
                    
                    print(f"DEBUG: Found potential new target at {new_target} with influence {new_influence}")
                else:
                    print("DEBUG: No valid positions found in grid after applying masks")
                    new_target = None
                    new_influence = None
            except Exception as e:
                print(f"DEBUG ERROR finding best influence position: {e}")
                new_target = None
                new_influence = None
            
            # Fall back to traditional method if grid-based approach failed
            if new_target is None:
                new_target = self.select_best_target(enemy_units, friendly_units, combined_mask)
            
            # Re-register our current target if we temporarily unregistered it
            if current_target_temporarily_unregistered and self.best_target_pos:
                try:
                    registered = nova_manager.register_nova_target(self.best_target_pos)
                    if registered:
                        print(f"DEBUG: Re-registered target at {self.best_target_pos}")
                    else:
                        print(f"DEBUG: Failed to re-register target at {self.best_target_pos}")
                except Exception as e:
                    print(f"DEBUG ERROR re-registering target: {e}")
            
            # If we found a significantly better target, switch to it
            if new_target and self.best_target_pos:
                try:
                    # Get the influence at the current target position
                    current_y, current_x = int(self.best_target_pos.y), int(self.best_target_pos.x)
                    current_influence = grid[current_y, current_x]
                    
                    # For the new target, use the influence we found if available
                    if new_influence is None:
                        new_y, new_x = int(new_target.y), int(new_target.x)
                        new_influence = grid[new_y, new_x]
                    
                    # Store the influence value for future comparisons
                    self.best_target_influence = new_influence
                    
                    # Calculate how much better the new target is (as a percentage)
                    # Use abs to handle negative influence values properly
                    improvement = (new_influence - current_influence) / max(abs(current_influence), 1)
                    
                    # Only switch if the improvement is significant (>10%)
                    if improvement > 0.1:
                        # Update our target
                        print(f"DEBUG: Updating Nova target from {self.best_target_pos} to {new_target} (improvement: {improvement:.2f})")
                        self.best_target_pos = new_target
                        
                        # Try to register the new target
                        try:
                            registered = nova_manager.register_nova_target(new_target)
                            if not registered:
                                print(f"DEBUG: Failed to register new target at {new_target}, but will still move there")
                        except Exception as e:
                            print(f"DEBUG ERROR registering new target: {e}")
                        return True
                    else:
                        print(f"DEBUG: New target improvement ({improvement:.2f}) below threshold, keeping current target")
                except Exception as e:
                    print(f"DEBUG ERROR calculating improvement: {e}")
            
            return False
            
        except Exception as e:
            print(f"DEBUG ERROR in update_target_position: {e}")
            return False

    def execute(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Attempt to execute Disruptor Nova ability. Initializes nova state and simulates firing the nova.
        Returns self (the nova instance) if fired successfully, or None if not.
        
        Args:
            disruptor_unit: The Disruptor unit to fire the Nova
            enemy_units: List of enemy units to target
            friendly_units: List of friendly units to avoid
            nova_manager: Optional NovaManager for target coordination
        """
        # Check if ability can be used (already handles cooldown via SC2 API)
        if not self.can_use(disruptor_unit):
            print(f"DEBUG: Disruptor {disruptor_unit.tag} cannot use Nova - ability not ready")
            return None

        # Get exclusion mask from nova manager if provided
        exclusion_mask = None
        if nova_manager:
            try:
                # Fresh call to get the grid
                grid = self.get_grid()
                if grid is None:
                    print("DEBUG: Tactical grid is None in execute")
                    return None
                
                # Then get the exclusion mask using the grid
                exclusion_mask = nova_manager.get_exclusion_mask(grid)
                print(f"DEBUG: Got exclusion mask with {np.sum(exclusion_mask)} cells excluded")
            except Exception as e:
                print(f"DEBUG ERROR getting exclusion mask: {e}")
                exclusion_mask = None

        # Select a target, considering exclusion zones
        print(f"DEBUG: Selecting target for Disruptor {disruptor_unit.tag}, {len(enemy_units)} enemy units, {len(friendly_units)} friendly units")
        target = None
        try:
            target = self.select_best_target(enemy_units, friendly_units, exclusion_mask)
        except Exception as e:
            print(f"DEBUG ERROR in select_best_target: {e}")
            
        if not target:
            print(f"DEBUG: No valid target found for Disruptor {disruptor_unit.tag}")
            return None
        else:
            print(f"DEBUG: Found target at {target}")
            
        # Register this target with the nova manager if provided
        target_registered = False
        if nova_manager:
            try:
                target_registered = nova_manager.register_nova_target(target)
                if not target_registered:
                    print(f"DEBUG: Failed to register target with NovaManager - continuing anyway")
                else:
                    print(f"DEBUG: Successfully registered target with NovaManager")
            except Exception as e:
                print(f"DEBUG ERROR registering target: {e}")
                # Continue anyway - we'll still try to fire the nova

        # Calculate which ability ID to use based on unit type (should be AbilityId.EFFECT_PURIFICATIONNOVA)
        ability_id = AbilityId.EFFECT_PURIFICATIONNOVA

        # Attempt to use the Nova ability
        did_fire = False
        try:
            did_fire = disruptor_unit(ability_id, target)
        except Exception as e:
            print(f"DEBUG ERROR firing Nova: {e}")
            
        print(f"DEBUG: Disruptor execute result: {did_fire}")
        
        if did_fire:
            # On successful fire, initialize the nova instance and add to active novas
            self.best_target_pos = target
            self.frames_left = 48  # 2.1 seconds duration at 22.4 frames per sec
            # TODO: We'll need to track the actual unit in a real game
            self.unit = None  # This would be assigned with the actual Nova unit
            return self
        else:
            # If firing failed, unregister the target only if we registered it successfully
            if nova_manager and target_registered:
                try:
                    nova_manager.unregister_nova_target(target)
                except Exception as e:
                    print(f"DEBUG ERROR unregistering unused target: {e}")
            return None

    def run_step(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Execute one step for this nova. Reduces the frame counter and possibly moves the nova.
        Returns True if the nova is still active, False if it has expired.
        
        Args:
            enemy_units: List of enemy units to avoid hitting
            friendly_units: List of friendly units to avoid hitting
            nova_manager: Optional NovaManager for target updating
        """
        if self.frames_left <= 0:
            return False

        # Update the frame counter
        self.frames_left -= 1
        
        # Check if we should update our target - do this every frame to be more responsive
        if nova_manager:
            self.update_target_position(enemy_units, friendly_units, nova_manager)

        # If a valid target was found and it differs from the current position, command the nova to move
        if self.best_target_pos is not None and self.best_target_pos != self.unit.position:
            #TODO need to find a way for nova to pathfind to the best target position
            self.unit.move(self.best_target_pos)
            print(f"Moving Nova to target: {self.best_target_pos}")

        # Continue running until the nova expires
        return self.frames_left > 0
        
    def load_info(self, unit):
        """Initialize nova tracking state when fired."""
        self.unit = unit
        self.frames_left = 48  # Reset to starting frame count
        self.distance_left = self.calculate_distance_left(unit.movement_speed)
        self.best_target_pos = unit.position  # Initial target position

    def update_info(self):
        """Update nova tracking state each step."""
        self.frames_left -= 1  # Decrement frame count
        self.distance_left = self.calculate_distance_left(self.unit.movement_speed)

    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate remaining distance based on unit movement speed and frames left.

        Uses the constant 22.4 frames per second to convert speed into distance per frame.
        """
        return self.frames_left * (unit_speed / 22.4)

    def run_debug(self):
        """Output debugging information."""
        print(f"Nova Frames Left: {self.frames_left}, Distance Left: {self.distance_left}")
        print(f"Current Position: {self.unit.position}, Target Position: {self.best_target_pos}")
        
        # Skip visualization if influence grid isn't available
        if not hasattr(self, 'influence_grid'):
            print("No influence grid available for visualization")
            return
        
        try:
            # Get max and min values to understand the range of influence
            max_value = np.max(self.influence_grid)
            min_value = np.min(self.influence_grid)
            print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
            
            # Create enemy influence visualization (values > 200)
            enemy_grid = self.influence_grid.copy()
            enemy_grid[enemy_grid <= 200] = 0  # Zero out non-enemy areas
            enemy_grid[enemy_grid > 200] -= 200  # Normalize to positive values starting from 0
            
            # Create friendly influence visualization (values < 200)
            friendly_grid = self.influence_grid.copy()
            friendly_grid[friendly_grid >= 200] = 0  # Zero out non-friendly areas
            friendly_grid = 200 - friendly_grid  # Invert for visualization (make small values large)
            friendly_grid[friendly_grid <= 0] = 0  # Remove any negative values
            
            print(f"Enemy influence - Max: {np.max(enemy_grid) if np.max(enemy_grid) > 0 else 0}")
            print(f"Friendly influence - Max: {np.max(friendly_grid) if np.max(friendly_grid) > 0 else 0}")
            
            # Draw enemy influence (green)
            if np.max(enemy_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=enemy_grid,
                    lower_threshold=10,  # Show significant enemy influence
                    upper_threshold=50,  # Cap for better visualization
                    color=(0, 255, 0)    # Green for enemy influence
                )
            
            # Draw friendly influence (red)
            if np.max(friendly_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=friendly_grid,
                    lower_threshold=10,   # Show significant friendly influence
                    upper_threshold=50,   # Cap for better visualization
                    color=(255, 0, 0)     # Red for friendly influence
                )
                
        except Exception as e:
            print(f"Error in grid visualization: {e}")

    def _visualize_grid(self, grid: np.ndarray) -> None:
        """
        Debug visualization method to show grid influence values.
        
        Args:
            grid: The tactical ground grid
        """
        try:
            print(f"DEBUG _visualize_grid: Visualizing grid with shape {grid.shape}")
            
            # Visualize the grid using debug boxes
            # This is just an example - adjust to your SC2 client's debug options
            
            # Assuming grid is a 2D array, find the highest influence points
            highest_points = []
            
            # Find the top 10 highest influence points
            flat_indices = np.argsort(grid.flatten())[-10:]  # Top 10 indices
            
            for flat_idx in flat_indices:
                # Convert flat index to 2D indices
                i, j = np.unravel_index(flat_idx, grid.shape)
                # Calculate game coordinates (assuming grid indices map directly)
                pos = Point2((j, i))
                value = grid[i, j]
                highest_points.append((pos, value))
                
            # Draw debug boxes at these points
            for pos, value in highest_points:
                # Color based on value - for example, red for high influence
                color = (255, 0, 0)  # Red
                # Example debug visualization
                if hasattr(self.bot.client, 'debug_box_out'):
                    # Draw a box at position
                    size = 1.0  # Size of box
                    p1 = Point3((pos.x - size/2, pos.y - size/2, 10))  # Bottom corner
                    p2 = Point3((pos.x + size/2, pos.y + size/2, 11))  # Top corner
                    self.bot.client.debug_box_out(p1, p2, color=color)
                    
                    # Also add text
                    text_pos = Point3((pos.x, pos.y, 12))
                    self.bot.client.debug_text_world(f"Influence: {value:.1f}", text_pos, color=color, size=10)
                    
            # Send the debug command
            if hasattr(self.bot.client, 'send_debug'):
                self.bot.client.send_debug()
                
            print(f"DEBUG _visualize_grid: Visualization completed for {len(highest_points)} points")
                
        except Exception as e:
            print(f"DEBUG ERROR in _visualize_grid: {e}")

    def get_grid(self) -> Optional[np.ndarray]:
        """
        Safely get the tactical ground grid.
        
        Returns:
            Optional[np.ndarray]: The tactical ground grid or None if there was an error
        """
        try:
            # Print the type of the mediator's get_tactical_ground_grid attribute
            print(f"DEBUG: Type of mediator.get_tactical_ground_grid: {type(self.mediator.get_tactical_ground_grid)}")

            # Get the grid (already an ndarray, not a method)
            result = self.mediator.get_tactical_ground_grid
            
            # Verify we have a valid ndarray
            if isinstance(result, np.ndarray):
                print(f"DEBUG: Got grid with shape {result.shape}")
                return result.copy()  # Return a copy to be safe
            else:
                print(f"DEBUG: get_tactical_ground_grid returned invalid type: {type(result)}")
                return None
        except Exception as e:
            print(f"DEBUG ERROR in get_grid: {e}")
            return None
