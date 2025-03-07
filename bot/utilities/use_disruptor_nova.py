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
import math

from cython_extensions import cy_all_points_below_max_value


if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, mediator: ManagerMediator, bot: 'AresBot'):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = 21.4
        self.nova_duration = 2.1
        self.best_target_pos = None
        self.best_target_influence = 0
        self.unit = None
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
        
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """
        Select the best target position for the nova.
        Uses the tactical grid to find optimal target areas.
        
        Args:
            disruptor_unit: The Disruptor unit that will fire the Nova
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            nova_manager: Optional NovaManager for target coordination
            
        Returns:
            The Point2 target position or None if no valid target found
        """
        if not enemy_units:
            print("DEBUG select_best_target: No enemy units to target")
            return None
            
        try:
            # Define the maximum search radius to limit computational load
            MAX_SEARCH_RADIUS = 15.0  # Units beyond this distance won't be considered
            
            # Filter units by distance to the disruptor to improve performance
            if disruptor_unit:
                disruptor_pos = disruptor_unit.position
                nearby_enemies = [unit for unit in enemy_units if unit.position.distance_to(disruptor_pos) <= MAX_SEARCH_RADIUS]
                
                print(f"DEBUG select_best_target: Filtered {len(nearby_enemies)} nearby enemies out of {len(enemy_units)} total")
                
                if not nearby_enemies:
                    print("DEBUG select_best_target: No nearby enemy units to target")
                    return None
            else:
                # If no disruptor unit is provided, use all units (less efficient)
                nearby_enemies = enemy_units
            
            # Get the tactical grid from the mediator
            grid = self.mediator.get_tactical_ground_grid
            
            if grid is None:
                print("DEBUG select_best_target: Tactical grid is None")
                return None
            
            # Find the position with the highest influence value
            # We'll check each enemy position and a small area around them
            best_pos = None
            best_influence = float('-inf')
            
            for enemy in nearby_enemies:
                try:
                    # Check the enemy position and a small area around it
                    enemy_pos = enemy.position
                    
                    # Check grid value at enemy position
                    try:
                        # Convert game position to grid indices
                        grid_x = int(enemy_pos.x)
                        grid_y = int(enemy_pos.y)
                        
                        # Get influence value at this position
                        if 0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]:
                            influence = grid[grid_y, grid_x]
                            
                            # Update best position if this is better
                            if influence > best_influence:
                                best_influence = influence
                                best_pos = enemy_pos
                                print(f"DEBUG: Found better position at {best_pos} with influence {best_influence}")
                    except Exception as e:
                        print(f"DEBUG ERROR checking grid at position {enemy_pos}: {e}")
                except Exception as e:
                    print(f"DEBUG ERROR processing enemy unit: {e}")
            
            # If we found a good position, return it
            if best_pos is not None:
                # Store for future comparisons
                self.best_target_pos = best_pos
                self.best_target_influence = best_influence
                
                print(f"DEBUG select_best_target: Found target at {best_pos} with influence {best_influence}")
                return best_pos
            else:
                print("DEBUG select_best_target: No suitable target found")
                return None
        
        except Exception as e:
            print(f"DEBUG ERROR in select_best_target: {e}")
            return None

    def _target_closest_enemy(self, enemy_units):
        """Simple fallback to target the closest enemy unit when grid-based targeting fails.
        
        Args:
            enemy_units: List of enemy units to consider
            
        Returns:
            Point2 target position or None if no enemies
        """
        # This method is no longer used - kept for reference
        print("DEBUG: fallback targeting no longer used")
        return None

    def _select_target_position_based(self, enemy_units, friendly_units, exclusion_mask=None):
        """Legacy position-based targeting as a fallback when grid-based targeting fails.
        DEPRECATED: This is kept for backward compatibility but should be removed in future updates.
        """
        print("DEBUG: Legacy position-based targeting no longer used")
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
                
                # Create a mask of positions that are too far to reach
                distance_mask = np.zeros_like(grid, dtype=bool)
                
                # Use the unit's position as the center of our search radius
                grid_center_x, grid_center_y = int(current_position.x), int(current_position.y)
                
                # For each point in the grid, calculate if it's too far from current position
                grid_height, grid_width = grid.shape
                for y in range(grid_height):
                    for x in range(grid_width):
                        # Calculate distance in game units
                        distance = ((x - grid_center_x) ** 2 + (y - grid_center_y) ** 2) ** 0.5
                        # Mark as True (excluded) if beyond max travel distance
                        if distance > max_travel_distance:
                            distance_mask[y, x] = True
                
                # Get the exclusion mask for other active Nova targets
                exclusion_mask = None
                try:
                    if nova_manager:
                        exclusion_mask = nova_manager.get_exclusion_mask(grid, ignore_position=self.best_target_pos)
                except Exception as e:
                    print(f"DEBUG ERROR getting exclusion mask in update: {e}")
                
                # Make a working copy of the grid
                working_grid = grid.copy()
                
                # Create a combined mask (both distance and exclusion)
                combined_mask = distance_mask.copy()
                if exclusion_mask is not None and exclusion_mask.shape == combined_mask.shape:
                    combined_mask = np.logical_or(combined_mask, exclusion_mask)
                
                # Apply combined mask - set excluded areas to a low value
                working_grid[combined_mask] = 100.0  # Low value (friendly influence)
                
                # Find best position (highest influence)
                new_y, new_x = np.unravel_index(np.argmax(working_grid), working_grid.shape)
                new_influence = working_grid[new_y, new_x]
                print(f"DEBUG update_target: Best candidate at ({new_x}, {new_y}) with influence {new_influence:.1f}")
                
                # Only use positions with at least some enemy influence
                if new_influence > 200:  # Values > 200 indicate enemy influence
                    new_pos = Point2((float(new_x), float(new_y)))
                    
                    # Compare with current target
                    should_update = False
                    
                    # Update if the new influence is significantly better
                    influence_improvement_threshold = 50.0
                    if new_influence > self.best_target_influence + influence_improvement_threshold:
                        should_update = True
                        print(f"DEBUG update_target: New influence {new_influence:.1f} is better than current {self.best_target_influence:.1f}")
                    
                    if should_update:
                        # Unregister old target
                        if nova_manager:
                            nova_manager.unregister_nova_target(self.best_target_pos)
                            
                        # Update target
                        old_pos = self.best_target_pos
                        self.best_target_pos = new_pos
                        self.best_target_influence = new_influence
                        
                        # Register new target
                        if nova_manager:
                            success = nova_manager.register_nova_target(new_pos)
                            print(f"DEBUG update_target: {'Successfully' if success else 'Failed to'} register new target")
                        
                        print(f"DEBUG update_target: Updated target from {old_pos} to {new_pos}")
                        return True
                
                return False
                
        except Exception as e:
            print(f"DEBUG ERROR in update_target_position: {e}")
            return False

    def execute(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """
        Execute the Nova ability.
        
        Args:
            disruptor_unit: The Disruptor unit
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            nova_manager: The NovaManager instance
            
        Returns:
            bool: True if the ability was used, False otherwise
        """
        try:
            print(f"DEBUG: Using nova_manager: {nova_manager is not None}")
            
            # Safety check for disruptor unit
            if disruptor_unit is None:
                print("DEBUG: No disruptor unit provided")
                return False
                
            # Check if unit is already casting Nova
            is_casting = False
            if hasattr(disruptor_unit, 'orders'):
                is_casting = any(order.ability.id == 390 for order in disruptor_unit.orders)
                    
            if is_casting:
                print(f"DEBUG: Disruptor unit already casting")
                return False
                
            # Initialize with appropriate nova manager
            if nova_manager:
                nova_manager.enemy_units = enemy_units
                nova_manager.friendly_units = friendly_units
                
            # Get target
            if not self.best_target_pos:
                # Find best target if we don't have one
                self.select_best_target(disruptor_unit, enemy_units, friendly_units, nova_manager)
                if not self.best_target_pos:
                    # Couldn't find a good target
                    print("DEBUG: No valid target found")
                    return False
                    
            # Check if we're already casting the ability
            if self.executing:
                print(f"DEBUG: Already executing Nova ability at {self.best_target_pos}")
                if self.unit is None:
                    print("DEBUG: Nova unit is None but executing flag is set, resetting")
                    self.reset()
                    return False
                    
                return True
                
            # Calculate distance to target
            try:
                target_distance = disruptor_unit.position.distance_to(self.best_target_pos)
            except Exception as e:
                print(f"DEBUG ERROR calculating distance: {e}")
                return False
            
            # Max travel distance for Nova (game units)
            nova_range = 13.0  # Maximum reasonable range for Nova ability
            if nova_manager and hasattr(nova_manager, 'nova_speed') and hasattr(nova_manager, 'nova_lifetime'):
                max_travel_distance = min(nova_range, nova_manager.nova_speed * nova_manager.nova_lifetime)
            else:
                max_travel_distance = nova_range
            
            print(f"Target distance: {target_distance:.2f}, Max travel distance: {max_travel_distance:.2f}")
            
            # Check for targets out of range
            if target_distance > max_travel_distance + 5.0:  # Add a small buffer for pathing
                print(f"Target out of range. Moving Disruptor to: {disruptor_unit.position.towards(self.best_target_pos, 3.0)}")
                try:
                    disruptor_unit.move(disruptor_unit.position.towards(self.best_target_pos, 3.0))
                except Exception as e:
                    print(f"DEBUG ERROR moving disruptor: {e}")
                return False
            
            # Target is in range, use ability
            print(f"Using Purification Nova at {self.best_target_pos}")
            try:
                disruptor_unit(AbilityId.EFFECT_PURIFICATIONNOVA, self.best_target_pos)
            except Exception as e:
                print(f"DEBUG ERROR casting Nova: {e}")
                return False
            
            # Mark as executing and set the Nova unit
            self.executing = True
            self.frames_left = int(nova_manager.nova_lifetime * 22.4) if nova_manager and hasattr(nova_manager, 'nova_lifetime') else 48
            self.unit = disruptor_unit
            
            return True
        except Exception as e:
            print(f"DEBUG ERROR executing Nova: {e}")
            
            # Clean up resources in case of error
            if self.best_target_pos and nova_manager:
                print(f"DEBUG: Unregistering Nova target at {self.best_target_pos} after error")
                try:
                    nova_manager.unregister_nova_target(self.best_target_pos)
                except Exception as unregister_error:
                    print(f"DEBUG ERROR unregistering target: {unregister_error}")
            
            self.reset()
            return False

    def reset(self):
        """Reset the Nova state."""
        self.best_target_pos = None
        self.best_target_influence = 0
        self.target_pos = None
        self.executing = False
        self.frames_left = 0
        self.unit = None

    def run_step(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Execute one step for this nova. Reduces the frame counter and possibly moves the nova.
        Args:
            enemy_units: List of enemy units to consider
            friendly_units: List of friendly units to avoid
            nova_manager: Optional NovaManager for target coordination
            
        Returns:
            bool: True if the Nova is still active, False if it has finished
        """
        if not self.executing:
            return False
            
        try:
            # Decrement frame counter
            self.frames_left -= 1
            
            # Update target if time remains
            if self.frames_left > 10 and nova_manager:
                try:
                    self.update_target_position(enemy_units, friendly_units, nova_manager)
                except Exception as e:
                    print(f"DEBUG ERROR updating target position: {e}")
            
            # Check if Nova has expired
            if self.frames_left <= 0:
                print("DEBUG: Nova expired, cleaning up")
                # Clean up on finished Nova
                if self.best_target_pos and nova_manager:
                    try:
                        nova_manager.unregister_nova_target(self.best_target_pos)
                    except Exception as e:
                        print(f"DEBUG ERROR unregistering expired Nova target: {e}")
                self.reset()
                return False
                
            # Check if Nova unit is still tracked
            if self.unit is None:
                print("DEBUG: Nova unit is no longer tracked")
                # Clean up if Nova was destroyed or removed
                if self.best_target_pos and nova_manager:
                    try:
                        nova_manager.unregister_nova_target(self.best_target_pos)
                    except Exception as e:
                        print(f"DEBUG ERROR unregistering destroyed Nova target: {e}")
                self.reset()
                return False
                
            return True
            
        except Exception as e:
            print(f"DEBUG ERROR in run_step: {e}")
            if self.best_target_pos and nova_manager:
                try:
                    nova_manager.unregister_nova_target(self.best_target_pos)
                except Exception as unregister_error:
                    print(f"DEBUG ERROR unregistering target: {unregister_error}")
            self.reset()
            return False
        
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
        
        # For debugging, let's get a fresh grid if we don't have one stored
        grid = self.mediator.get_tactical_ground_grid
        if grid is not None:
            self.influence_grid = grid.copy()
            print(f"Successfully obtained grid with shape {grid.shape}")
        else:
            print("No grid available for visualization")
            return
        
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

    def _visualize_grid(self, grid: np.ndarray) -> None:
        """
        Debug visualization method to show grid influence values.
        
        Args:
            grid: The tactical ground grid
        """
        try:
            print(f"DEBUG _visualize_grid: Visualizing grid with shape {grid.shape}")
            
            # Store the grid for visualization
            self.influence_grid = grid.copy()
            
            # Get max and min values to understand the range of influence
            max_value = np.max(grid[~np.isinf(grid)])
            min_value = np.min(grid[~np.isinf(grid)])
            print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
            
            # Create enemy influence visualization (values > 200)
            enemy_grid = grid.copy()
            enemy_grid[np.isinf(enemy_grid)] = 0
            enemy_grid[enemy_grid <= 200] = 0  # Zero out non-enemy areas
            enemy_grid[enemy_grid > 200] -= 200  # Normalize to positive values starting from 0
            
            # Create friendly influence visualization (values < 200)
            friendly_grid = grid.copy()
            friendly_grid[np.isinf(friendly_grid)] = 0
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
                
            print(f"DEBUG _visualize_grid: Visualization completed")
                
        except Exception as e:
            print(f"DEBUG ERROR in _visualize_grid: {e}")
