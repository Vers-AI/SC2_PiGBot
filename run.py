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
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer



import yaml

from bot.bot import PiG_Bot
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
    # attempt to get race and bot name from config file if they exist
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file)
            if MY_BOT_NAME in config:
                bot_name = config[MY_BOT_NAME]
            if MY_BOT_RACE in config:
                race = Race[config[MY_BOT_RACE].title()]

    bot1 = Bot(race, PiG_Bot(), bot_name)

    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        
        map_list: List[str] = [
            "AcropolisAIE",
            "AutomatonAIE",
            "EphemeronAIE",
            "InterloperAIE",
            "ThunderbirdAIE"
        ]

        random_race = Race.Terran #random.choice([Race.Zerg, Race.Terran, Race.Protoss])
        print("Starting local game...")
        run_game(
            maps.get(random.choice(map_list)),
            [
                bot1,
                Computer(Race.Zerg, Difficulty.Medium, ai_build=AIBuild.Rush),
            ],
            realtime=False,
        )


# Start game
if __name__ == "__main__":
    main()