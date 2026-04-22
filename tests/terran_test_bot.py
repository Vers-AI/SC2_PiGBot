"""Terran Test Bot — Modular Terran opponent for testing threat detection.

Purpose: Controlled Terran opponent with selectable behavior modules.
         Each module tests a specific threat type that PiGBot needs to handle.
Key Decisions: Minimal economy, behavior modules are toggled via class attributes.
Limitations: No defense, no upgrades beyond what modules require.

Usage: --TestBot=Terran (default: all modules active)
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


class TerranTestBot(BotAI):
    """Terran test bot with modular threat behaviors.

    Set class attributes to enable/disable specific test modules:
      - enable_widow_mines: Build and burrow Widow Mines (default True)
      - enable_siege_tanks: Siege any spawned tanks (default True)
    """

    enable_widow_mines: bool = False
    enable_siege_tanks: bool = True

    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        cc = self.townhalls.first
        if cc is None:
            return

        # --- Shared economy ---
        await self._run_economy(cc)

        # --- Behavior modules ---
        if self.enable_widow_mines:
            await self._run_widow_mines(cc)
        if self.enable_siege_tanks:
            await self._run_siege_tanks()

    async def _run_economy(self, cc) -> None:
        """Minimal Terran economy: SCVs, supply, refinery."""
        if cc.is_idle and self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
            cc.train(UnitTypeId.SCV)

        if self.supply_left < 4 and not self.already_pending(UnitTypeId.SUPPLYDEPOT):
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                await self.build(UnitTypeId.SUPPLYDEPOT, near=cc.position.towards(self.game_info.map_center, 5))

        if not self.structures.of_type(UnitTypeId.REFINERY) and self.can_afford(UnitTypeId.REFINERY):
            vgs = self.vespene_geyser.closer_than(15, cc)
            if vgs:
                await self.build(UnitTypeId.REFINERY, near=vgs.first)

        for scv in self.workers.idle:
            if self.structures.of_type(UnitTypeId.REFINERY).ready:
                ref = self.structures.of_type(UnitTypeId.REFINERY).ready.first
                if ref.surplus_harvesters < 0:
                    scv.gather(ref)
                    continue
            scv.gather(self.mineral_field.closest_to(scv))

    async def _run_widow_mines(self, cc) -> None:
        """Widow Mine module: build Factory+TechLab, produce and burrow mines."""
        # Build Barracks (prerequisite for Factory)
        if not self.structures.of_type(UnitTypeId.BARRACKS) and self.can_afford(UnitTypeId.BARRACKS):
            await self.build(UnitTypeId.BARRACKS, near=cc.position.towards(self.game_info.map_center, 8))

        # Build Factory
        if (
            self.structures.of_type(UnitTypeId.BARRACKS).ready
            and not self.structures.of_type(UnitTypeId.FACTORY)
            and self.can_afford(UnitTypeId.FACTORY)
        ):
            await self.build(UnitTypeId.FACTORY, near=cc.position.towards(self.game_info.map_center, 10))

        # Attach Tech Lab to Factory
        factory = self.structures.of_type(UnitTypeId.FACTORY).ready
        if factory:
            fac = factory.first
            if not fac.has_add_on and fac.is_idle and self.can_afford(UnitTypeId.FACTORYTECHLAB):
                fac.build(UnitTypeId.FACTORYTECHLAB)

        # Produce Widow Mines
        fac_with_lab = self.structures.of_type(UnitTypeId.FACTORY).ready.filter(lambda f: f.has_add_on)
        if fac_with_lab:
            fac = fac_with_lab.first
            if fac.is_idle and self.can_afford(UnitTypeId.WIDOWMINE) and self.supply_left > 0:
                fac.train(UnitTypeId.WIDOWMINE)

        # Burrow mines near enemy start
        if self.enemy_start_locations:
            enemy_start = self.enemy_start_locations[0]
            for mine in self.units(UnitTypeId.WIDOWMINE).idle:
                dist = mine.distance_to(enemy_start)
                if dist < 20:
                    mine(AbilityId.BURROWDOWN_WIDOWMINE)
                else:
                    mine.move(enemy_start.towards(mine.position, 15))

    async def _run_siege_tanks(self) -> None:
        """Siege Tank module: siege up any unsieged tanks (spawned via debug)."""
        for tank in self.units(UnitTypeId.SIEGETANK):
            tank(AbilityId.SIEGEMODE_SIEGEMODE)