# PiG_Bot - Bits of Code Bot

Welcome to PiG_Bot, the example bot showcased in the "Bits of Code" series! This bot is designed as a starting point for those looking to dive into the world of AI bot development in StarCraft II using Python-SC2 and Ares framework.

## Episode Feature
PiG_Bot was featured in an episode of "Bits of Code," which you can watch [here](https://www.youtube.com/playlist?list=PLN2WDx0iwG9V2BehVgv-tg_U0OcrWAdKP). The episode walks you through the bot's strategy of mimicking the StarCraft commentator/player PiG, using his Bronze to GM strategies.

## Strategy
PiG_Bot's strategy is inspired by PiG's Bronze to GM series:
1. Mimic PiG's strategies and decision-making processes.
2. Adapt to various in-game situations using pre-defined strategies.

## How to Use PiG_Bot
You can deconstruct PiG_Bot or modify it as you wish
1. Clone the repository:
```bash
git clone --recursive <your_git_repo_home_url_here>
```
2. Navigate to the root folder:
```bash
cd <bot_folder>
```
3. Install Poetry:
```bash
poetry install
```
4. If you have a non-standard StarCraft 2 installation or are using Linux, adjust `MAPS_PATH` in `run.py`.
5. Run PiG_Bot in a StarCraft II AI match:
```bash
poetry run python run.py
```

## Template
You can get the original template from [VersAI SC2 Bot Template](https://github.com/Vers-AI/versusai-sc2-bot-template).

## License
This project is released under the [MIT License](LICENSE).
