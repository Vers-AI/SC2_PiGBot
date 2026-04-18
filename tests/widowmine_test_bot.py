"""Widow Mine Test Bot — Terran bot that builds and fires Widow Mines.

Purpose: Controlled opponent for testing threat-detection diagnostics.
         Builds Widow Mines, burrows them near the enemy, and lets them fire.
Key Decisions: Minimal economy (just enough to produce mines), mines auto-fire.
Limitations: No micro, no defense, no upgrades. Pure mine production.
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


class WidowMineTestBot(BotAI):
    """Terran bot that mass-produces Widow Mines and burrows them."""

    async def on_step(self, iteration: int):
        if iteration == 0:
            # Initial build: split workers
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        # Constantly produce SCVs from CC
        cc = self.townhalls.first
        if cc and cc.is_idle and self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
            cc.train(UnitTypeId.SCV)

        # Build supply when needed
        if self.supply_left < 4 and not self.already_pending(UnitTypeId.SUPPLYDEPOT):
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                await self.build(UnitTypeId.SUPPLYDEPOT, near=cc.position.towards(self.game_info.map_center, 5))

        # Build Refinery on first Vespene Geyser
        if not self.structures.of_type(UnitTypeId.REFINERY) and self.can_afford(UnitTypeId.REFINERY):
            vgs = self.vespene_geyser.closer_than(15, cc)
            if vgs:
                await self.build(UnitTypeId.REFINERY, near=vgs.first)

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

        # Attach Tech Lab to Factory once complete
        factory = self.structures.of_type(UnitTypeId.FACTORY).ready
        if factory:
            fac = factory.first
            if not fac.has_add_on and fac.is_idle and self.can_afford(UnitTypeId.FACTORYTECHLAB):
                fac.build(UnitTypeId.FACTORYTECHLAB)

        # Produce Widow Mines from Factory with Tech Lab
        fac_with_lab = self.structures.of_type(UnitTypeId.FACTORY).ready.filter(lambda f: f.has_add_on)
        if fac_with_lab:
            fac = fac_with_lab.first
            if fac.is_idle and self.can_afford(UnitTypeId.WIDOWMINE) and self.supply_left > 0:
                fac.train(UnitTypeId.WIDOWMINE)

        # Burrow Widow Mines near enemy start location
        if self.enemy_start_locations:
            enemy_start = self.enemy_start_locations[0]
            for mine in self.units(UnitTypeId.WIDOWMINE).idle:
                # Move toward enemy start, then burrow when close
                dist = mine.distance_to(enemy_start)
                if dist < 20:
                    # Close enough — burrow here
                    mine(AbilityId.BURROWDOWN_WIDOWMINE)
                else:
                    # Move closer
                    mine.move(enemy_start.towards(mine.position, 15))

        # Rally SCVs to mine when we have a refinery
        for scv in self.workers.idle:
            if self.structures.of_type(UnitTypeId.REFINERY).ready:
                ref = self.structures.of_type(UnitTypeId.REFINERY).ready.first
                if ref.surplus_harvesters < 0:
                    scv.gather(ref)
                    continue
            scv.gather(self.mineral_field.closest_to(scv))