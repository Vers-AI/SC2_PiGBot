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
        For each iteration this method goes through one enemy unit, then goes through every other unit.
        
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
            # Make a fresh call to get the tactical grid each time - we'll use this as a fallback
            grid = self.get_grid()
            if grid is None:
                print("DEBUG select_best_target: Warning - tactical grid is None")
                # Continue anyway, we'll use a position-based approach
            else:
                print(f"DEBUG select_best_target: Grid info - shape={grid.shape}, min={np.min(grid)}, max={np.max(grid)}")
                
                # Apply the exclusion mask if provided
                if exclusion_mask is not None:
                    try:
                        # Make sure the shapes match
                        if grid is not None and exclusion_mask.shape != grid.shape:
                            print(f"DEBUG select_best_target: Warning - exclusion mask shape {exclusion_mask.shape} doesn't match grid shape {grid.shape}")
                            exclusion_mask = None
                        else:
                            # Apply exclusion mask - if grid is None, we'll just use the mask for positions
                            if grid is not None:
                                grid = np.where(exclusion_mask, 0, grid)
                                print(f"DEBUG select_best_target: Applied exclusion mask, min={np.min(grid)}, max={np.max(grid)}")
                    except Exception as e:
                        print(f"DEBUG ERROR applying exclusion mask: {e}")

            # NEW APPROACH: Directly use enemy positions to find clusters
            print(f"DEBUG select_best_target: Using position-based targeting with {len(enemy_units)} enemy units")
            
            # If no enemy units, early exit
            if not enemy_units:
                return None
                
            # 1. Calculate a targeting score for various points on the map
            # We'll create a grid of potential target positions
            resolution = 8  # Lower = higher resolution but more computation
            
            # Create a list of potential target positions
            if grid is not None:
                # Use grid dimensions to determine boundaries
                grid_height, grid_width = grid.shape
                x_min, y_min = 0, 0 
                x_max, y_max = grid_width, grid_height
            else:
                # Without a grid, use estimated map boundaries (approximately a 200x200 square)
                x_min, y_min = 0, 0
                x_max, y_max = 200, 200
            
            # Generate candidate positions
            candidate_positions = []
            scores = []

            # First, include enemy positions and their surroundings as candidates
            for enemy in enemy_units:
                pos = enemy.position
                candidate_positions.append(pos)
                
                # Also add some surrounding positions for better coverage
                for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                    candidate_positions.append(Point2((pos.x + dx, pos.y + dy)))
                
            # Add a systematic grid of positions for broader coverage
            for x in range(x_min, x_max, resolution):
                for y in range(y_min, y_max, resolution):
                    candidate_positions.append(Point2((x, y)))
            
            # 2. Score each position based on enemy clustering
            for pos in candidate_positions:
                # Skip if this position is in an excluded area and we have a mask
                if exclusion_mask is not None and grid is not None:
                    try:
                        # Convert position to grid indices (assuming grid coordinates match game coordinates)
                        grid_y = min(max(int(pos.y), 0), grid.shape[0] - 1)
                        grid_x = min(max(int(pos.x), 0), grid.shape[1] - 1)
                        
                        if exclusion_mask[grid_y, grid_x]:
                            continue  # Skip this position as it's excluded
                    except Exception as e:
                        print(f"DEBUG ERROR checking exclusion at {pos}: {e}")
                
                # Calculate potential damage at this position
                # Score is the number of enemy units within Nova radius (1.5)
                nova_radius = 1.5
                enemies_hit = 0
                for enemy in enemy_units:
                    if pos.distance_to(enemy.position) <= nova_radius:
                        enemies_hit += 1
                
                # Penalize for friendly fire
                friendly_hit = 0
                for friendly in friendly_units:
                    if pos.distance_to(friendly.position) <= nova_radius:
                        friendly_hit += 1
                
                # Only consider positions that hit at least one enemy and don't hit friendlies
                if enemies_hit > 0 and friendly_hit == 0:
                    # Score = number of enemies hit
                    scores.append((pos, enemies_hit))
            
            # Sort by score (highest first)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores:
                best_pos, hit_count = scores[0]
                print(f"DEBUG select_best_target: Selected target at {best_pos} hitting {hit_count} enemies")
                
                # Store the best target position and its value for other methods
                self.best_target_pos = best_pos
                self.best_target_influence = hit_count * 100  # Convert hit count to influence-like value
                
                # Debug visualization if desired
                if self.should_debug_visuals and grid is not None:
                    self._visualize_grid(grid)
                
                return best_pos
            else:
                print("DEBUG select_best_target: No valid targets found based on enemy positions")
                return None
            
        except Exception as e:
            print(f"DEBUG Error in select_best_target: {e}")
            return None

    def update_target_position(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager) -> bool:
        """
        Check if a better target has become available within the Nova's remaining travel range.
        
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
            if self.frames_left < 10:  # Don't change course if almost expired
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
            
            # Create a mask for areas out of reach
            out_of_reach_mask = np.ones(grid.shape, dtype=bool)
            
            # Mark areas within reach as False (not excluded)
            height, width = grid.shape
            for y in range(height):
                for x in range(width):
                    # Convert grid coordinates to game coordinates
                    game_pos = Point2((x, y))
                    
                    # Calculate distance from Nova to this position
                    distance = current_position.distance_to(game_pos)
                    
                    # If within reach, mark as not excluded
                    if distance <= max_travel_distance:
                        out_of_reach_mask[y, x] = False
            
            # Get the exclusion mask from the nova manager
            try:
                exclusion_mask = nova_manager.get_exclusion_mask(grid)
                combined_mask = np.logical_or(exclusion_mask, out_of_reach_mask)
            except Exception as e:
                print(f"ERROR getting exclusion mask: {e}")
                combined_mask = out_of_reach_mask
            
            # Only keep the exclusion around our current target
            if self.best_target_pos:
                try:
                    nova_manager.unregister_nova_target(self.best_target_pos)
                except Exception as e:
                    print(f"ERROR unregistering target: {e}")
            
            # Find the new best target
            new_target = self.select_best_target(enemy_units, friendly_units, combined_mask)
            
            # Re-register our current target regardless of outcome
            if self.best_target_pos:
                try:
                    nova_manager.register_nova_target(self.best_target_pos)
                except Exception as e:
                    print(f"ERROR re-registering target: {e}")
            
            # If we found a significantly better target, switch to it
            if new_target and self.best_target_influence > 0:
                # Calculate the influence at the old position 
                try:
                    old_influence = grid[int(self.best_target_pos.y), int(self.best_target_pos.x)]
                    
                    # Calculate how much better the new target is (as a percentage)
                    improvement = (self.best_target_influence - old_influence) / self.best_target_influence
                    
                    # Only switch if the improvement is significant (>20%)
                    if improvement > 0.2:
                        # Update our target
                        self.best_target_pos = new_target
                        # Try to register the new target
                        try:
                            nova_manager.register_nova_target(new_target)
                        except Exception as e:
                            print(f"ERROR registering new target: {e}")
                        return True
                except Exception as e:
                    print(f"ERROR calculating improvement: {e}")
            
            return False
            
        except Exception as e:
            print(f"Error in update_target_position: {e}")
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
        
        # Check if we should update our target (only every 5 frames to reduce computational load)
        if nova_manager and self.frames_left % 5 == 0:
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
