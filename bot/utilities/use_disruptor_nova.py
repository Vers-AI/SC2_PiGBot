"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.ability_id import AbilityId


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.behaviors.combat.individual import UseAbility
from ares.managers.manager_mediator import ManagerMediator
from bot.utilities.get_nova_aoe_grid import get_nova_aoe_grid, apply_influence_in_radius
from ares.dicts.unit_data import UNIT_DATA
import time

if TYPE_CHECKING:
    from ares import AresBot

class DummyNovaUnit:
    """A dummy class to simulate a nova projectile unit. """
    def __init__(self, position, movement_speed=5.0):
        self.position = position
        self.movement_speed = movement_speed

    def move(self, target):
        print(f"[DummyNovaUnit] Moving from {self.position} to {target}")
        self.position = target


class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, cooldown: float, nova_duration: float, map_data, bot):
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

    def can_use(self, current_time: float) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (current_time - self.last_used) >= self.cooldown

    def select_best_target(self, enemy_units: list, friendly_units: list) -> tuple:
        """Determine the best target position based on grid influence data."""
        # Assume self.map_data provides the required map information and grid
        grid = self.map_data

        # Initialize the influence grid
        influence_grid = get_nova_aoe_grid(grid)

        # Apply influence from enemy and friendly units using map_data in apply_influence_in_radius
        for enemy in enemy_units:
            army_value = UNIT_DATA[enemy.UnitID]['army_value'] if enemy.UnitID in UNIT_DATA else 0
            influence_grid = apply_influence_in_radius(influence_grid, (enemy.position.x, enemy.position.y), radius=enemy.radius, influence=army_value, map_data=self.map_data)
        for friendly in friendly_units:
            army_value = UNIT_DATA[friendly.UnitID]['army_value'] if friendly.UnitID in UNIT_DATA else 0
            influence_grid = apply_influence_in_radius(influence_grid, (friendly.position.x, friendly.position.y), radius=friendly.radius, influence=-army_value, map_data=self.map_data)

        # Find the position with the maximum influence
        max_influence = float('-inf')
        best_position = None
        for i in range(influence_grid.shape[0]):
            for j in range(influence_grid.shape[1]):
                if influence_grid[i][j] > max_influence:
                    max_influence = influence_grid[i][j]
                    best_position = (i, j)

        # Convert grid coordinates to world coordinates
        if best_position:
            cell_size = 1.0  # Assume a cell size
            best_position = (best_position[0] * cell_size, best_position[1] * cell_size)

        return best_position

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

    def run_step(self, enemy_units: list, friendly_units: list):
        """Update best target position based on grid influence data and command the nova to move accordingly."""
        # Use the grid influence data to select the best target position
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

    def execute(self, disruptor_unit, enemy_units: list, friendly_units: list, current_time: float):
        """Attempt to execute Disruptor Nova ability. Initializes nova state and simulates firing the nova.
        Returns self (the nova instance) if fired successfully, or None if not.
        """
        print("Executing Disruptor Nova...")
        if not self.can_use(current_time):
            return None

        # Use select_best_target method to choose an initial target
        target = self.select_best_target(enemy_units, friendly_units)
        if target is None:
            return None
        self.bot.register_behavior(
            UseAbility(
                AbilityId.EFFECT_PURIFICATIONNOVA, disruptor_unit, target
            )
        )
        print(f"Firing Disruptor Nova at target: {target}")

        # Simulate nova creation (in a real scenario, this would come from the game API)
        nova_unit = self.nova_creation(disruptor_unit, target)

        # Initialize nova tracking with the nova unit
        self.load_info(nova_unit)

        # Record the time the nova was fired
        self.last_used = current_time
        return self

    def nova_creation(self, disruptor_unit, target):
        """Wait for the nova creation event by polling the NovaManager for a nova unit near the target position."""
        timeout = 5  # seconds
        poll_interval = 0.1  # seconds
        elapsed_time = 0

        while elapsed_time < timeout:
            # Poll the NovaManager for a nova unit near the target
            for nova in self.bot.nova_manager.get_active_novas():
                if nova.position.distance_to(target) < 1.0:  # Assuming a small threshold for proximity
                    return nova
            time.sleep(poll_interval)
            elapsed_time += poll_interval

        print("Warning: Nova creation event timed out.")
        return None