"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from sc2.ids.ability_id import AbilityId

import traceback

from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.behaviors.combat.individual import PathUnitToTarget
from ares.managers.manager_mediator import ManagerMediator
import numpy as np
from sc2.position import Point2
import math

from cython_extensions import cy_all_points_below_max_value


if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, mediator: ManagerMediator, bot: 'AresBot', position_update_frequency: int = 10):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = 21.4
        self.nova_duration = 2.1
        self.best_target_pos = None
        self.best_target_influence = 0
        self.unit = None
        self.original_target_pos = None
        self.target_pos = None
        self.executing = False
        self.frames_left = 0
        self.distance_left = 0.0
        self.mediator = mediator
        self.bot = bot  # Store bot instance
        self.should_debug_visuals = False  # Set to True for visual debugging
        
        # Do NOT store any references to grids here, get them fresh when needed
        self.influence_grid = None
        
        # Print initialization message
        print("DEBUG: UseDisruptorNova initialized")
        
        # Add movement cooldown tracking
        self.last_movement_frame = 0
        self.movement_cooldown = 5  # Only issue move commands every 5 frames
        
        # Add a timestamp to track when this Nova was started
        self.execution_start_time = 0
        self.max_execution_time = 10.0  # Maximum time in seconds a Nova should be executing
        
        # Position tracking and movement management
        self.max_distance_tracking_factor = 35.0  # Maximum distance to consider for tracking by the Nova
        self.frame_counter = 0  # To track frames since execution for position updates
        self.position_update_frequency = position_update_frequency  # How often to update position
        self.current_position = None  # Current Nova position
        self.initial_unit_position = None  # Position when the Nova was created
        self.last_target_update_time = 0  # Last time we updated the target position
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units, friendly_units, exclusion_mask=None):
        """
        Select the best target position for the nova based on the provided units.
        Uses influence values to find optimal target areas while considering enemy clusters
        and avoiding friendly fire.
        
        Args:
            enemy_units: Units collection of enemy units to consider as targets
            friendly_units: Units collection of friendly units to avoid damaging
            exclusion_mask: Optional mask of areas to exclude from targeting
            
        Returns:
            The Point2 target position or None if no valid target found
        """
        try:
            # Check if we have enemy units to target
            if not enemy_units or len(enemy_units) == 0:
                print("DEBUG select_best_target: No enemy units to target")
                return None
                
            # Get the tactical grid from the mediator
            grid = self.mediator.get_tactical_ground_grid
            
            if grid is None:
                print("DEBUG select_best_target: Tactical grid is None")
                return None
                
            # Log exclusion mask status if provided
            if exclusion_mask is not None:
                try:
                    excluded_cells = np.sum(exclusion_mask)
                    print(f"DEBUG select_best_target: Using exclusion mask with {excluded_cells} cells excluded")
                except Exception as e:
                    print(f"DEBUG ERROR in exclusion mask sum: {e}")
            
            # Define Nova radius for targeting
            nova_radius = 1.5  # Units within this radius will be hit
            
            # Create candidate positions from enemy units and their surroundings
            candidate_positions = []
            
            # First, include enemy positions and their surroundings as candidates
            for enemy in enemy_units:
                try:
                    pos = enemy.position
                    candidate_positions.append(pos)
                    
                    # Also add some surrounding positions for better coverage
                    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                        candidate_positions.append(Point2((pos.x + dx, pos.y + dy)))
                except Exception as e:
                    print(f"DEBUG ERROR adding candidate position: {e}")
            
            # Score each candidate position
            best_pos = None
            best_score = float('-inf')
            best_influence = float('-inf')
            
            for pos in candidate_positions:
                try:
                    # Skip if this position is in an exclusion zone
                    if exclusion_mask is not None:
                        try:
                            # Convert position to grid indices
                            grid_x = int(pos.x)
                            grid_y = int(pos.y)
                            
                            if (0 <= grid_y < exclusion_mask.shape[0] and 
                                0 <= grid_x < exclusion_mask.shape[1] and 
                                exclusion_mask[grid_y, grid_x]):
                                continue  # Skip this position as it's excluded
                        except Exception as e:
                            print(f"DEBUG ERROR checking exclusion at {pos}: {e}")
                    
                    # Count enemies within Nova radius
                    enemies_hit = 0
                    for enemy in enemy_units:
                        if pos.distance_to(enemy.position) <= nova_radius:
                            enemies_hit += 1
                    
                    # Check for friendly fire
                    friendly_hit = 0
                    for friendly in friendly_units:
                        if pos.distance_to(friendly.position) <= nova_radius:
                            friendly_hit += 1
                    
                    # Skip positions that would hit friendly units
                    if friendly_hit > 0:
                        continue
                    
                    # Get influence value at this position
                    influence = 0
                    try:
                        # Convert position to grid indices
                        grid_x = int(pos.x)
                        grid_y = int(pos.y)
                        
                        if (0 <= grid_y < grid.shape[0] and 
                            0 <= grid_x < grid.shape[1]):
                            influence = grid[grid_y, grid_x]
                    except Exception as e:
                        print(f"DEBUG ERROR getting influence at {pos}: {e}")
                    
                    # Calculate final score: combine influence and enemy count
                    # Weight enemy count more heavily to prioritize clusters
                    score = (enemies_hit * 100) + influence
                    
                    # Update best position if this is better
                    if score > best_score:
                        best_score = score
                        best_influence = influence
                        best_pos = pos
                        print(f"DEBUG select_best_target: Found better position at {best_pos} with score {best_score} (influence: {influence}, enemies: {enemies_hit})")
                except Exception as e:
                    print(f"DEBUG ERROR evaluating position {pos}: {e}")
            
            # If we found a good position, return it
            if best_pos is not None:
                # Store for future comparisons
                self.best_target_pos = best_pos
                self.best_target_influence = best_influence
                
                print(f"DEBUG select_best_target: Selected target at {best_pos} with score {best_score}")
                return best_pos
            else:
                print("DEBUG select_best_target: No suitable target found")
                return None
        
        except Exception as e:
            print(f"DEBUG ERROR in select_best_target: {e}")
            return None

    def update_target_position(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager):
        """
        Check if a better target has become available within Nova's remaining travel range.
        
        Args:
            enemy_units: List of enemy units to consider
            friendly_units: List of friendly units to avoid
            nova_manager: The NovaManager for exclusion mask and target registration
            
        Returns:
            bool: True if target was updated, False otherwise
        """
        if not self.best_target_pos or not self.unit:
            return False
            
        try:
            # Get the tactical grid
            grid = self.mediator.get_tactical_ground_grid
            if grid is None:
                print("DEBUG: Tactical grid is None in update_target_position")
                return False
            
            # Calculate maximum distance the Nova can still travel based on remaining frames
            if nova_manager and hasattr(nova_manager, 'nova_speed'):
                # Calculate max travel distance based on remaining frames
                frames_per_second = 22.4  # SC2 runs at 22.4 frames per second
                remaining_seconds = self.frames_left / frames_per_second
                max_travel_distance = nova_manager.nova_speed * remaining_seconds
                
                # Current Nova position is our starting point
                current_position = self.unit.position
                
                print(f"DEBUG update_target: {self.frames_left} frames left, max travel distance: {max_travel_distance:.2f}")
                
                # Filter enemy units by distance to the Nova unit
                MAX_SEARCH_RADIUS = max_travel_distance + 2.0  # Add a small buffer
                nearby_enemies = [unit for unit in enemy_units if unit.position.distance_to(current_position) <= MAX_SEARCH_RADIUS]
                
                if not nearby_enemies:
                    print("DEBUG update_target: No nearby enemy units within reach")
                    return False
                
                # Get the exclusion mask for other active Nova targets
                exclusion_mask = None
                try:
                    if nova_manager:
                        # Get exclusion mask but ignore our current target
                        exclusion_mask = nova_manager.get_exclusion_mask(grid)
                        
                        # Modify the mask to not exclude our current target
                        if exclusion_mask is not None and self.best_target_pos:
                            # Convert target position to grid indices
                            target_x, target_y = int(self.best_target_pos.x), int(self.best_target_pos.y)
                            
                            # Create a small area around our current target that's allowed
                            if 0 <= target_y < exclusion_mask.shape[0] and 0 <= target_x < exclusion_mask.shape[1]:
                                radius = int(nova_manager.exclusion_radius)
                                y_min = max(0, target_y - radius)
                                y_max = min(exclusion_mask.shape[0], target_y + radius + 1)
                                x_min = max(0, target_x - radius)
                                x_max = min(exclusion_mask.shape[1], target_x + radius + 1)
                                
                                # Allow our current target area
                                exclusion_mask[y_min:y_max, x_min:x_max] = False
                except Exception as e:
                    print(f"DEBUG ERROR getting exclusion mask in update: {e}")
                
                # Find the position with the highest influence value among nearby enemies
                best_pos = None
                best_influence = float('-inf')
                
                for enemy in nearby_enemies:
                    try:
                        # Check the enemy position
                        enemy_pos = enemy.position
                        
                        # Check if this position is reachable
                        if not nova_manager.can_nova_reach_target(current_position, enemy_pos, self.frames_left):
                            continue
                        
                        # Check grid value at enemy position
                        try:
                            # Convert game position to grid indices
                            grid_x = int(enemy_pos.x)
                            grid_y = int(enemy_pos.y)
                            
                            # Skip if this position is in an exclusion zone
                            if exclusion_mask is not None:
                                if exclusion_mask[grid_y, grid_x]:
                                    continue
                            
                            # Get influence value at this position
                            influence = grid[grid_y, grid_x]
                            
                            # Use a very low threshold to accept almost any target
                            # We'd rather have some target than none
                            if influence > best_influence:
                                best_influence = influence
                                best_pos = enemy_pos
                                print(f"DEBUG update_target: Found better position at {best_pos} with influence {best_influence}")
                        except Exception as e:
                            print(f"DEBUG ERROR getting influence value: {e}")
                    except Exception as e:
                        print(f"DEBUG ERROR processing enemy unit: {e}")
                
                # If we found a good position, return it
                if best_pos is not None:
                    # Only update if the new influence is significantly better or the position is significantly different
                    influence_improvement_threshold = 50.0
                    position_change_threshold = 3.0
                    
                    should_update = False
                    
                    # Update if the new influence is significantly better
                    if best_influence > self.best_target_influence + influence_improvement_threshold:
                        should_update = True
                        print(f"DEBUG update_target: New influence {best_influence:.1f} is better than current {self.best_target_influence:.1f}")
                    
                    # Or update if the position is significantly different
                    elif self.best_target_pos.distance_to(best_pos) > position_change_threshold:
                        should_update = True
                        print(f"DEBUG update_target: New position {best_pos} is significantly different from current {self.best_target_pos}")
                    
                    if should_update:
                        # Unregister old target
                        if nova_manager:
                            nova_manager.unregister_nova_target(self.best_target_pos)
                            
                        # Update target
                        old_pos = self.best_target_pos
                        self.best_target_pos = best_pos
                        self.best_target_influence = best_influence
                        
                        # Register new target
                        if nova_manager:
                            success = nova_manager.register_nova_target(best_pos)
                            print(f"DEBUG update_target: {'Successfully' if success else 'Failed to'} register new target")
                        
                        print(f"DEBUG update_target: Updated target from {old_pos} to {best_pos}")
                        return True
                
                return False
                
        except Exception as e:
            print(f"DEBUG ERROR in update_target_position: {e}")
            return False

    def load_info(self, nova_unit) -> None:
        """Load information about the nova unit."""
        # Store unit reference
        self.unit = nova_unit
        self.current_position = nova_unit.position
        
        # Only initialize frames_left if this is a new execution (not already executing)
        if not self.executing:
            self.frames_left = round(2.1 * 22.4)  # ~2.1 seconds at 22.4 frames per second
            print(f"DEBUG: Initializing Nova with {self.frames_left} frames to live")
            # Set executing flag
            self.executing = True
            # Reset the frame counter
            self.frame_counter = 0
        else:
            print(f"DEBUG: Nova already executing with {self.frames_left} frames left, not resetting")
        
        # Initialize movement tracking
        self.previous_position = nova_unit.position
        self.consecutive_stuck_frames = 0
        
        # Initialize time-based tracking
        

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

        # Get exclusion mask from nova_manager if provided
        exclusion_mask = None
        if nova_manager:
            try:
                # Fresh call to get the grid
                grid = self.mediator.get_tactical_ground_grid
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
            target = self.select_best_target(enemy_units, friendly_units, exclusion_mask, nova_manager)
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

    
    def select_best_target_pos(self, disruptor_unit, enemy_units, friendly_units=None, nova_manager=None) -> Point2:
        """
        Select the best position to target with the Nova.
        
        Args:
            disruptor_unit: The disruptor unit to use the nova with
            enemy_units: List of enemy units to consider as targets
            friendly_units: (Optional) List of friendly units to avoid damaging
            nova_manager: (Optional) The NovaManager instance
            
        Returns:
            Point2: The best target position, or None if no valid target
        """
        try:
            # Convert to empty lists if None
            if enemy_units is None:
                enemy_units = []
            if friendly_units is None:
                friendly_units = []
                
            # Filter out non-attackable units
            valid_enemy_units = [
                unit for unit in enemy_units
                if unit.is_visible and not unit.is_flying and not unit.is_structure
            ]
            
            if not valid_enemy_units:
                print(f"DEBUG: No valid enemy units to target with Nova")
                return None
                
            # Filter enemies that are in range
            nova_range = 15.0  # Maximum reasonable range for Nova ability
            in_range_enemies = [
                unit for unit in valid_enemy_units
                if unit.distance_to(disruptor_unit) < nova_range
            ]
            
            if not in_range_enemies:
                print(f"DEBUG: No enemy units in range of disruptor")
                return None
                
            # Find the best cluster of enemies
            best_enemy = None
            best_score = -1
            
            for enemy in in_range_enemies:
                # Calculate base score from distance to disruptor (closer is better)
                # Use inverse distance so closer enemies have higher scores
                distance_to_disruptor = enemy.distance_to(disruptor_unit)
                if distance_to_disruptor < 1.0:
                    distance_to_disruptor = 1.0  # Avoid division by zero
                
                base_score = 100.0 / distance_to_disruptor
                
                # Count nearby enemies to this enemy for cluster score
                nearby_count = sum(1 for other in in_range_enemies if other.distance_to(enemy) < 2.5)
                cluster_score = nearby_count * 50.0  # Each nearby enemy adds 50 points
                
                # Combine scores
                total_score = base_score + cluster_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_enemy = enemy
            
            # If we found a good target, return its position
            if best_enemy:
                return best_enemy.position
                
            return None
            
        except Exception as e:
            print(f"DEBUG ERROR selecting best target for Nova: {e}")
            return None

    def update_info(self):
        """Update nova tracking state each step."""
        self.frames_left -= 1  # Decrement frame count
        self.distance_left = self.calculate_distance_left(self.unit.movement_speed)
    
    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate remaining distance based on unit movement speed and frames left.

        Uses the constant 22.4 frames per second to convert speed into distance per frame.
        """
        return self.frames_left * (unit_speed / 22.4)

    
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

    
    

    def run_debug(self):
        """Output debugging information."""
        print(f"Nova Frames Left: {self.frames_left}, Distance Left: {self.distance_left}")
        print(f"Current Position: {self.unit.position if hasattr(self.unit, 'position') else self.last_known_position}, Target Position: {self.best_target_pos}")
        
        # For debugging, let's get a fresh grid if we don't have one stored
        self.influence_grid = self.mediator.get_tactical_ground_grid
        
        try:
            # Get max and min values to understand the range of influence
            max_value = np.max(self.influence_grid)
            min_value = np.min(self.influence_grid)
            
            # Calculate percentiles for better understanding of distribution
            valid_grid = self.influence_grid[~np.isinf(self.influence_grid)]
            if len(valid_grid) > 0:
                p10 = np.percentile(valid_grid, 10)
                p25 = np.percentile(valid_grid, 25)
                p50 = np.percentile(valid_grid, 50)  # median
                p75 = np.percentile(valid_grid, 75)
                p90 = np.percentile(valid_grid, 90)
                
                # Count values in different ranges
                below150 = np.sum(valid_grid < 150)
                range150_190 = np.sum((valid_grid >= 150) & (valid_grid < 190))
                range190_210 = np.sum((valid_grid >= 190) & (valid_grid <= 210))
                range210_250 = np.sum((valid_grid > 210) & (valid_grid <= 250))
                above250 = np.sum(valid_grid > 250)
                
                print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
                print(f"Grid percentiles - 10%: {p10:.1f}, 25%: {p25:.1f}, 50%: {p50:.1f}, 75%: {p75:.1f}, 90%: {p90:.1f}")
                print(f"Value distribution - Below 150: {below150}, 150-190: {range150_190}, 190-210: {range190_210}, 210-250: {range210_250}, Above 250: {above250}")
            else:
                print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
                print("No valid (non-infinite) values in grid for percentile analysis")
            
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