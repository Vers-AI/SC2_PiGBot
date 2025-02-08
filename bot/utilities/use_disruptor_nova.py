"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""

import time

class UseDisruptorNova:
    def __init__(self, cooldown: float, nova_duration: float):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = cooldown
        self.nova_duration = nova_duration
        self.last_used = -cooldown  # Allow immediate use on start

    def can_use(self, current_time: float) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (current_time - self.last_used) >= self.cooldown

    def select_best_target(self, enemy_units: list, friendly_units: list):
        """Select the best target for Disruptor Nova based on enemy and friendly positions.

        TODO: Implement grid influence logic to improve target selection.
        For now, a simple placeholder returns the first enemy target if available.
        """
        if enemy_units:
            return enemy_units[0]  # Naive selection; to be improved with grid influence
        return None

    def execute(self, disruptor_unit, enemy_units: list, friendly_units: list) -> bool:
        """Attempt to execute Disruptor Nova ability.

        Returns True if the ability was successfully executed, else False.
        """
        current_time = time.time()
        if not self.can_use(current_time):
            return False
        target = self.select_best_target(enemy_units, friendly_units)
        if target:
            # Fire Nova ability on the target.
            # Integration with the game API to issue the order should occur here.
            self.last_used = current_time
            # TODO: Add on-screen tracking logic for nova_duration
            print(f"Firing Disruptor Nova at target: {target}")
            return True
        return False