"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from sc2.ids.ability_id import AbilityId


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
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
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units: List['Unit'], friendly_units: List['Unit']) -> Optional[Point2]:
        """
        Select the best target for the Disruptor Nova based on enemy and friendly unit positions.
        
        Uses the tactical ground grid from ARES where values above 200 indicate enemy
        influence and values below 200 indicate friendly influence.
        
        Returns:
            Point2: The position to target, or None if no good target is found.
        """
        if not enemy_units:
            return None
        
        try:
            # Need to get a fresh grid each time this runs
            self.influence_grid = self.mediator.get_tactical_ground_grid
            
            # Create a copy of the influence grid for targeting analysis
            target_grid = self.influence_grid.copy()
            
            # Filter for areas with enemy influence and without friendly influence
            target_grid[target_grid < 200] = 0  # Reset areas with friendly influence to zero
            
            # Check if we have any viable targets
            if np.max(target_grid) <= 200:
                return None
                
            # Get the coordinates of the maximum influence
            max_indices = np.unravel_index(np.argmax(target_grid), target_grid.shape)
            
            # Convert indices to Point2
            best_position = Point2((max_indices[1], max_indices[0]))  # Swap coordinates (column, row) to (x, y)
            
            # Find closest enemy to the best position
            closest_enemy = None
            min_distance = float('inf')
            for enemy in enemy_units:
                dist = enemy.position.distance_to(best_position)
                if dist < min_distance:
                    min_distance = dist
                    closest_enemy = enemy
                    
            if closest_enemy:
                return closest_enemy.position
        except Exception as e:
            print(f"Error finding target: {e}")
            
        return None

    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate remaining distance based on unit movement speed and frames left.

        Uses the constant 22.4 frames per second to convert speed into distance per frame.
        """
        return self.frames_left * (unit_speed / 22.4)

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

    def run_step(self, enemy_units: List['Unit'], friendly_units: List['Unit']):
        """Update best target position based on enemy clustering and command the nova to move accordingly."""
        # Use the clustering data to select the best target position
        self.best_target_pos = self.select_best_target(enemy_units, friendly_units)

        # If a valid target was found and it differs from the current position, command the nova to move
        if self.best_target_pos is not None and self.best_target_pos != self.unit.position:
            self.unit.move(self.best_target_pos)
            print(f"Moving Nova to target: {self.best_target_pos}")

        # Optionally, output debug information
        self.run_debug()

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

    def execute(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit']):
        """Attempt to execute Disruptor Nova ability. Initializes nova state and simulates firing the nova.
        Returns self (the nova instance) if fired successfully, or None if not.
        """
        # Check if ability can be used
        if not self.can_use(disruptor_unit):
            return None

        # Select a target
        target = self.select_best_target(enemy_units, friendly_units)
        if not target:
            return None

        # Send the command to fire the nova ability at the target
        disruptor_unit(AbilityId.EFFECT_PURIFICATIONNOVA, target)

        # Debug info
        print(f"Firing Disruptor Nova at target: {target}")

        # Return self to indicate success
        return self
