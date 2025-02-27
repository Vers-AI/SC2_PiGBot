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
from ares.behaviors.combat.individual import UseAbility
from ares.managers.manager_mediator import ManagerMediator
from bot.utilities.get_nova_aoe_grid import get_nova_aoe_grid, apply_influence_in_radius
from ares.dicts.unit_data import UNIT_DATA
import time
import numpy as np
from sc2.position import Point2

if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, cooldown: float, nova_duration: float, map_data, bot: 'AresBot'):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = cooldown
        self.nova_duration = nova_duration
        self.last_used = -cooldown  # Allow immediate use on start
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
        """Determine the best target position based on grid influence data."""
        if not enemy_units:
            return None
        
        # Calculate influence grid
        influence_grid = get_nova_aoe_grid(self.map_data)
        
        # Apply enemy influence
        for enemy in enemy_units:
            # Determine army value
            army_value = UNIT_DATA[enemy.type_id]['army_value'] if enemy.type_id in UNIT_DATA else 1
            
            # Game coordinates to cell coordinates
            # Find closest enemy with highest army value cluster
            for friendly in friendly_units:
                # Apply negative influence for friendly units to avoid friendly fire
                apply_influence_in_radius(
                    grid=influence_grid,
                    center=(enemy.position.x, enemy.position.y),
                    radius=2.5,  # Nova radius
                    influence=army_value,
                    map_data=self.map_data
                )
                
                # Apply negative influence for friendly units to avoid friendly fire
                apply_influence_in_radius(
                    grid=influence_grid,
                    center=(friendly.position.x, friendly.position.y),
                    radius=2.5,  # Nova radius
                    influence=-army_value * 2,  # Double negative to heavily discourage hitting friendlies
                    map_data=self.map_data
                )
        
        # Find the position with the maximum influence
        max_influence = np.max(influence_grid)
        if max_influence <= 0:
            return None
            
        # Get the coordinates of the maximum influence
        max_indices = np.unravel_index(np.argmax(influence_grid), influence_grid.shape)
        
        # Since grid coordinates roughly match the game coordinates, we can directly use them
        # If needed, adjust based on your map's resolution
        best_position = Point2((max_indices[0], max_indices[1]))
        
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

        # Simulate nova creation (in a real scenario, this would come from the game API)
        # TODO add a way to pass the nova unit into self.load_info after its created to steer it
        # Initialize nova tracking with the nova unit
        #self.load_info(nova_unit)

        # Record the time the nova was fired
        self.last_used = time.time()

        # Return self to indicate success
        return self
