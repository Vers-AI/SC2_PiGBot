"""Worker Rush Bot — sends all workers to attack enemy start location.

Purpose: Minimal opponent bot for testing.
Key Decisions: Uses idle probes first, falls back to all probes.
Limitations: No micro, no economy, just A-move workers.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId


class WorkerRushBot(BotAI):
    """A simple bot that rushes with workers to the enemy start location."""

    async def on_step(self, iteration: int):
        workers = self.units(UnitTypeId.PROBE).idle

        if not workers:
            workers = self.units(UnitTypeId.PROBE)

        if not workers:
            return

        if self.enemy_start_locations:
            enemy_start = self.enemy_start_locations[0]
        else:
            return

        for worker in workers:
            worker.attack(enemy_start)