"""
Standalone Wall Placement Generator Bot

Purpose: Generate natural wall placements for new maps and save to building_placements.yml
Usage: python generate_wall_placements.py

This bot does nothing except analyze the current map and generate wall coordinates.
It will overwrite existing map entries or append new ones as needed.
"""

import asyncio
import sys
import yaml
import logging
from pathlib import Path
from typing import Optional

# Add ARES to path (same as run.py)
sys.path.append('ares-sc2/src/ares')
sys.path.append('ares-sc2/src')
sys.path.append('ares-sc2')

from sc2 import maps
from sc2.main import run_game
from sc2.data import Difficulty, Race
from sc2.bot_ai import BotAI
from sc2.player import Bot, Computer
from ares import AresBot
from bot.utilities.natural_wall_manager import NaturalWallManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WallGeneratorBot(AresBot):
    """Minimal bot that only generates wall placements"""
    
    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)
        self.wall_manager = None
        
    async def on_start(self):
        """Initialize and generate wall placement on start"""
        # CRITICAL: Call superclass on_start for proper ARES initialization
        await super().on_start()
        
        logger.info("🏗️  Wall Generation Bot Starting...")
        
        # Initialize wall manager (after ARES is set up)
        self.wall_manager = NaturalWallManager(self)
        
        # Generate wall placement for current map
        map_name = self.game_info.map_name
        logger.info(f"📍 Analyzing map: {map_name}")
        
        # Generate for Zerg (most common opponent)
        opponent_race = "Zerg" 
        
        try:
            wall_positions = await self.wall_manager.generate_wall_placement(opponent_race)
            
            if wall_positions:
                logger.info(f"✅ Generated wall positions successfully!")
                self.save_wall_to_yaml(map_name, opponent_race, wall_positions)
                logger.info(f"💾 Saved wall placement to building_placements.yml")
            else:
                logger.error(f"❌ Failed to generate wall positions")
                
        except Exception as e:
            logger.error(f"❌ Error during generation: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Exit after generation
        logger.info("🏁 Wall generation complete - exiting")
        await self.client.leave()
        
    async def on_step(self, iteration: int):
        """Do nothing during game loop"""
        # CRITICAL: Call superclass on_step for proper ARES functionality
        await super().on_step(iteration)
        
        if iteration == 1:
            # Exit on first step to avoid running a full game
            await self.client.leave()
            
    def determine_spawn_position(self) -> str:
        """Determine if we're UpperSpawn or LowerSpawn based on map position"""
        our_start = self.start_location
        map_center = self.game_info.map_center
        
        # Simple approach: compare Y coordinate to map center
        if our_start.y > map_center.y:
            return "UpperSpawn"
        else:
            return "LowerSpawn"
    
    def save_wall_to_yaml(self, map_name: str, race: str, wall_positions):
        """Save wall data to building_placements.yml in ARES format"""
        yaml_path = Path("building_placements.yml")
        
        # Load existing data or create new
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Error loading existing YAML: {e}")
                data = {}
        else:
            data = {}
            
        # Ensure Protoss section exists
        if "Protoss" not in data:
            data["Protoss"] = {}
        
        # Use the exact map name from game metadata (what ARES will look for)
        clean_map_name = map_name
        
        # Determine spawn position
        spawn_position = self.determine_spawn_position()
        
        # Remove existing entry for this map if it exists (to overwrite)
        if clean_map_name in data["Protoss"]:
            logger.info(f"🔄 Overwriting existing data for {clean_map_name}")
            del data["Protoss"][clean_map_name]
        
        # Build wall data in exact ARES format matching the example
        wall_data = {
            "VsZergNatWall": {
                "AvailableVsRaces": ["Zerg", "Random"],
                spawn_position: {
                    "FirstPylon": wall_positions.first_pylon,
                    "Pylons": wall_positions.pylons, 
                    "ThreeByThrees": wall_positions.three_by_threes,
                    "StaticDefences": wall_positions.static_defences,
                    "GateKeeper": wall_positions.gate_keeper
                }
            }
        }
        
        # Add to data structure
        data["Protoss"][clean_map_name] = wall_data
        
        # Save back to file with proper formatting
        try:
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, width=1000)
            logger.info(f"💾 Successfully updated {yaml_path} with {clean_map_name} wall data")
        except Exception as e:
            logger.error(f"❌ Error saving YAML: {e}")

def main():
    """Run the wall generator bot"""
    
    # You can specify a map here or use the default
    map_name = "IncorporealAIE_v4"  # Change this to test different maps
    
    logger.info(f"🚀 Starting wall generation for {map_name}")
    
    try:
        run_game(
            maps.get(map_name),
            [
                Bot(Race.Protoss, WallGeneratorBot()),
                Computer(Race.Zerg, Difficulty.VeryEasy)  # Dummy opponent
            ],
            realtime=False,
            save_replay_as=None  # Don't save replays
        )
    except Exception as e:
        logger.error(f"❌ Error running wall generator: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
