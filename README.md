# PiG_Bot
A Human-Like StarCraft II Training Opponent

Built using [python-sc2](https://burnysc2.github.io/python-sc2/index.html) and the [ARES framework](https://aressc2.github.io/ares-sc2/api_reference/index.html).

## Project Overview

**What Inspired This Project?**  
The default StarCraft II in-game AI makes a terrible training partner. This project aims to provide a more challenging, human-like opponent for competitive players.

**Is This For You?**  
- StarCraft II players seeking a challenging AI at different skill levels.

**Why You'll Love This Bot**  
- The bot plays in a way that mimics human opponents.
- Difficulty is adjustable.
- The bot adapts to its opponent’s strategies.

**How You Can Use This Bot**  
- Play against the bot in custom games.
- Use it as a training partner to improve your skills.

**Key Features**  
- Builds according to the Bronze to GM macro standard.
- Scouts and gathers enemy information.
- Reacts to common cheese (Zergling Rush, Cannon Rush, Probe Rush).
- Uses disruptors and basic micro (a-move, stutter step).
- Responds to threats with appropriate unit counters.
- Maintains 3 bases and 66 probes.
- Switches builds based on scouting (pre-defined build switches).
- Fights only when favored.
- Runs 2 build orders per matchup:
    - **PvP:** 4 Gate All-in, 2 Gate Expand
    - **PvZ:** Standard Robo Opener, Double Stargate Phoenix (vs Mutas)
    - **PvT:** Standard Macro Build, Proxy Rax Response

**Tracked Data**  
- Opponent race, map name, game result, build order used, build loss flag, time build switched, enemy openings and units seen, air tech detection, fight outcomes, units lost, expansions taken, and more (see code for full list).

📋 **Community & Task List:** Join the discussion and see current tasks on the [VersusAI Community Forum](https://community.versusai.net/t/pig-bot/49)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- [StarCraft II](https://starcraft2.blizzard.com/en-us/) (free edition works fine)
- Poetry package manager

### Installation

1. Clone the repository:
```bash
git clone --recursive https://github.com/Vers-AI/SC2_PiGBot.git
```

2. Navigate to the root folder:
```bash
cd SC2_PiGBot
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Configure StarCraft II path (if needed):
   - If you have a non-standard StarCraft 2 installation or are using Linux, adjust `MAPS_PATH` in `run.py`

### Running the Bot

```bash
poetry run python run.py
```

## Project Structure
```
/bot/             # Core bot logic
├── /utilities/   # Helper functions and utilities
├── /managers/    # Bot behavior managers
/data/            # Game data and configs
/run.py           # Entry point to run the bot
```
## Contributing Guidelines

### How to Contribute
1. Check the [community forum thread](https://community.versusai.net/t/pig-bot/49) for current tasks
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide for Python code
- Write meaningful commit messages
- Add comments for complex logic
- Update documentation when changing functionality

## Resources

- [python-sc2 Documentation](https://burnysc2.github.io/python-sc2/index.html)
- [ARES Framework Documentation](https://aressc2.github.io/ares-sc2/api_reference/index.html)

## License

This project is released under the [MIT License](LICENSE).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

Want to contribute and be recognized? We invite you to join our AI Makers community: [https://versusai.net/join/](https://versusai.net/join/)

Community members get access to contribution opportunities, recognition, and more!

This project follows the [all-contributors](https://allcontributors.org) specification. Contributions of any kind welcome!
