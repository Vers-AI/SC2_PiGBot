"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.ability_id import AbilityId


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.managers.manager_mediator import ManagerMediator
from bot.utilities.get_nova_aoe_grid import get_nova_aoe_grid, apply_influence_in_radius
from ares.dicts.unit_data import UNIT_DATA
import time
import numpy as np
from sc2.position import Point2

if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, map_data, bot: 'AresBot'):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = 21.4
        self.nova_duration = 2.1
        self.best_target_pos = None
        self.frames_left = 48  # Example starting frame count
        self.distance_left = 0.0
        self.unit = None  # References the Purification Nova ability which is a unit not a spell
        self.map_data = map_data
        self.bot = bot  # Store bot instance
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units: List['Unit'], friendly_units: List['Unit']) -> Optional[Point2]:
        """
        Select the best target for the Disruptor Nova based on enemy and friendly unit positions.
        
        Uses an influence grid where enemy units add positive influence and friendly units add negative influence.
        The target position will be the location with maximum positive influence.
        
        Returns:
            Point2: The position to target, or None if no good target is found.
        """
        if not enemy_units:
            return None
        
        # Calculate influence grid
        self.influence_grid = get_nova_aoe_grid(self.map_data)
        
        # Apply enemy influence
        for enemy in enemy_units:
            # Determine army value
            army_value = UNIT_DATA[enemy.type_id]['army_value'] if enemy.type_id in UNIT_DATA else 1
            
            # Apply positive influence for enemy units
            apply_influence_in_radius(
                grid=self.influence_grid,
                center=(int(enemy.position.x), int(enemy.position.y)),  # Ensure integer coordinates
                radius=2.5,  # Nova radius
                influence=army_value,
                map_data=self.map_data
            )
        
        # Apply negative influence for friendly units (separate loop)
        for friendly in friendly_units:
            # Determine army value
            army_value = UNIT_DATA[friendly.type_id]['army_value'] if friendly.type_id in UNIT_DATA else 1
            
            # Apply negative influence for friendly units to avoid friendly fire
            apply_influence_in_radius(
                grid=self.influence_grid,
                center=(int(friendly.position.x), int(friendly.position.y)),  # Ensure integer coordinates
                radius=2.5,  # Nova radius
                influence=-army_value * 2,  # Double negative to heavily discourage hitting friendlies
                map_data=self.map_data
            )
                
        # Find the position with the maximum influence
        max_influence = np.max(self.influence_grid)
        if max_influence <= 0:
            return None
            
        # Get the coordinates of the maximum influence
        max_indices = np.unravel_index(np.argmax(self.influence_grid), self.influence_grid.shape)
        
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

        # Optionally, output debug information
        self.run_debug()

    def run_debug(self):
        """Output debugging information."""
        print(f"Nova Frames Left: {self.frames_left}, Distance Left: {self.distance_left}")
        print(f"Current Position: {self.unit.position}, Target Position: {self.best_target_pos}")
        
        # Get max and min values, excluding zeros
        non_zero_mask = self.influence_grid != 0
        if np.any(non_zero_mask):
            max_value = np.max(self.influence_grid[non_zero_mask])
            min_value = np.min(self.influence_grid[non_zero_mask])
            print(f"Max influence: {max_value}, Min influence: {min_value}")
        
        # Use a threshold that ignores zero values but shows both positive and negative influences
        # Positive threshold for enemy influence (green)
        self.bot.map_data.draw_influence_in_game(
            grid=self.influence_grid,
            lower_threshold=0.1,  # Show any positive influence above this threshold
            upper_threshold=50,   # Cap for better visualization
            color=(0, 255, 0)     # Green for positive influence
        )
        
        # Negative threshold for friendly unit influence (red)
        # Create a copy of the grid with only negative values
        negative_grid = self.influence_grid.copy()
        negative_grid[negative_grid >= 0] = 0  # Zero out non-negative values
        negative_grid = -negative_grid  # Invert for visualization (make negative values positive)
        
        # Draw the negative influence in red
        self.bot.map_data.draw_influence_in_game(
            grid=negative_grid,
            lower_threshold=0.1,  # Show any negative influence above this threshold (now positive after inversion)
            upper_threshold=50,   # Cap for better visualization
            color=(255, 0, 0)     # Red for negative influence
        )

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
