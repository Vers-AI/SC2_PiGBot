import sys
import random
import sys
from os import path
from pathlib import Path
from typing import List

sys.path.append('ares-sc2/src/ares')
sys.path.append('ares-sc2/src')
sys.path.append('ares-sc2')

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer, AbstractPlayer
from sc2.ids.unit_typeid import UnitTypeId
from sc2.units import Units

import yaml

from bot import PiG_Bot
from ladder import run_ladder_game

# change if non default setup / linux
# if having issues with this, modify `map_list` below manually
MAPS_PATH: str = "C:\\Program Files (x86)\\StarCraft II\\Maps"
CONFIG_FILE: str = "config.yml"
MAP_FILE_EXT: str = "SC2Map"
MY_BOT_NAME: str = "MyBotName"
MY_BOT_RACE: str = "MyBotRace"


def main():
    bot_name: str = "MyBot"
    race: Race = Race.Random

    __user_config_location__: str = path.abspath(".")
    user_config_path: str = path.join(__user_config_location__, CONFIG_FILE)
    
    # Load config and check for wall generation mode
    wall_generation_mode = False
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file)
            if MY_BOT_NAME in config:
                bot_name = config[MY_BOT_NAME]
            if MY_BOT_RACE in config:
                race = Race[config[MY_BOT_RACE].title()]
            # Check for wall generation mode
            if "WallGenerationMode" in config:
                wall_generation_mode = config["WallGenerationMode"]

    # Choose bot based on mode
    if wall_generation_mode:
        print("🏗️  Running in Wall Generation Mode!")
        # Import WallGeneratorBot only when needed (avoids import errors on ladder)
        from generate_wall_placements import WallGeneratorBot
        bot_instance = WallGeneratorBot()
        bot_name = "WallGenerator"
    else:
        print("🤖 Running normal PiG_Bot")
        bot_instance = PiG_Bot()
        
    bot1 = Bot(race, bot_instance, bot_name)

    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        
        map_list: List[str] = [
            "TorchesAIE_v4",
            "PylonAIE_v4",
            "PersephoneAIE_v4",
            "IncorporealAIE_v4",
            "LeyLinesAIE_v3",
            "UltraloveAIE_v2",
            "MagannathaAIE_v2"
        ]

        print("Starting local game...")
        run_game(
            maps.get(random.choice(map_list)),
            [
                bot1,
                Computer(Race.Terran, Difficulty.VeryHard, ai_build=AIBuild.Rush),
            ],
            realtime=False,
        )


class WorkerRushBot(BotAI):
    """A simple bot that rushes with workers to the enemy start location."""
    
    async def on_step(self, iteration: int):
        # Get all our workers
        workers = self.units(UnitTypeId.PROBE).idle
        
        # If we don't have any idle workers, try to get any workers
        if not workers:
            workers = self.units(UnitTypeId.PROBE)
            
        # If we still don't have any workers, we're probably dead
        if not workers:
            return  # Nothing to do
        
        if self.enemy_start_locations:
            enemy_start = self.enemy_start_locations[0]
        else:
            return  # No enemy location known yet
            
        # Send all workers to attack the enemy start location
        for worker in workers:
            worker.attack(enemy_start)


# Start game
if __name__ == "__main__":
    main()