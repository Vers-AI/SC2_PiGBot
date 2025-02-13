"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.managers.manager_mediator import ManagerMediator
from bot.utilities.get_nova_aoe_grid import get_nova_aoe_grid, apply_influence_in_radius

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
    def __init__(self, cooldown: float, nova_duration: float, map_data):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = cooldown
        self.nova_duration = nova_duration
        self.last_used = -cooldown  # Allow immediate use on start
        self.best_target_pos = None
        self.frames_left = 48  # Example starting frame count
        self.distance_left = 0.0
        self.unit = None  # References the Purification Nova ability which is a unit not a spell
        self.map_data = map_data

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
            influence_grid = apply_influence_in_radius(influence_grid, (enemy.position.x, enemy.position.y), radius=3, influence=5, map_data=self.map_data)
        for friendly in friendly_units:
            influence_grid = apply_influence_in_radius(influence_grid, (friendly.position.x, friendly.position.y), radius=3, influence=-5, map_data=self.map_data)

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

    def execute(self, disruptor_unit, enemy_units: list, friendly_units: list, current_time: float) -> bool:
        """Attempt to execute Disruptor Nova ability. Initializes nova state and simulates firing the nova."""
        if not self.can_use(current_time):
            return False

        # Use select_best_target method to choose an initial target
        target = self.select_best_target(enemy_units, friendly_units)
        if target is None:
            return False

        print(f"Firing Disruptor Nova at target: {target}")

        # Simulate nova creation (in a real scenario, this would come from the game API)
        nova_unit = self.simulate_nova_creation(disruptor_unit, target)

        # Initialize nova tracking with the nova unit
        self.load_info(nova_unit)

        # Record the time the nova was fired
        self.last_used = current_time
        return True

    def simulate_nova_creation(self, disruptor_unit, target):
        """Simulate the creation of a nova projectile unit. Returns a DummyNovaUnit instance."""
        return DummyNovaUnit(target)