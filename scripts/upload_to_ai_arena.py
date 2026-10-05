"""
Purpose: Upload bot.zip to AI Arena via its REST API.
Key Decisions: Wiki bio comes from bot_description.md; the field is omitted from the PATCH when that file is missing, so the site bio is never clobbered.
Limitations: Requires UPLOAD_API_TOKEN / UPLOAD_BOT_ID env vars and bot.zip in the repo root.
"""
from os import path, environ
from typing import Union

import requests
import yaml
from loguru import logger

API_TOKEN_ENV: str = "UPLOAD_API_TOKEN"
BOT_ID_ENV: str = "UPLOAD_BOT_ID"
CONFIG_FILE: str = "config.yml"
AUTO_UPLOAD_TO_AIARENA: str = "AutoUploadToAiarena"
BOT_ZIP_PUBLICLY_DOWNLOADABLE: str = "BotZipPubliclyDownloadable"
BOT_DATA_PUBLICLY_DOWNLOADABLE: str = "BotDataPubliclyDownloadable"
BOT_DATA_ENABLED: str = "BotDataEnabled"
DESCRIPTION_FILE: str = "bot_description.md"
ZIPFILE_NAME: str = "bot.zip"

TOKEN: str = environ.get(API_TOKEN_ENV)
BOT_ID: str = environ.get(BOT_ID_ENV)
URL: str = f"https://aiarena.net/api/bots/{BOT_ID}/"


def get_bot_description() -> Union[str, None]:
    """
    Read the bot bio from bot_description.md.
    Returns None when the file is missing so the AI Arena wiki
    is left untouched rather than overwritten with stale content.
    """
    description_path: str = path.join(path.abspath("."), DESCRIPTION_FILE)
    if not path.isfile(description_path):
        return None
    with open(description_path, encoding="utf-8") as description_file:
        return description_file.read()


def retrieve_value_from_config(string: str) -> Union[str, bool, None]:
    __user_config_location__: str = path.abspath(".")
    user_config_path: str = path.join(__user_config_location__, CONFIG_FILE)
    # attempt to get race and bot name from config file if they exist
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file)
            if string in config:
                return config[string]



if __name__ == "__main__":
    can_upload: bool = False
    if upload := retrieve_value_from_config(AUTO_UPLOAD_TO_AIARENA):
        can_upload = upload

    if not can_upload:
        logger.info(
            "Auto update to aiarena not enabled, please set "
            "AutoUploadToAiarena option in config to `True`"
        )

    else:
        logger.info("Uploading bot")
        
        # Read config values with defaults
        bot_zip_public = retrieve_value_from_config(BOT_ZIP_PUBLICLY_DOWNLOADABLE)
        if bot_zip_public is None:
            bot_zip_public = False
        
        bot_data_public = retrieve_value_from_config(BOT_DATA_PUBLICLY_DOWNLOADABLE)
        if bot_data_public is None:
            bot_data_public = False
        
        bot_data_enabled = retrieve_value_from_config(BOT_DATA_ENABLED)
        if bot_data_enabled is None:
            bot_data_enabled = True
        
        with open(ZIPFILE_NAME, "rb") as bot_zip:
            request_headers = {
                "Authorization": f"Token {TOKEN}",
            }
            request_data = {
                "bot_zip_publicly_downloadable": bot_zip_public,
                "bot_data_publicly_downloadable": bot_data_public,
                "bot_data_enabled": bot_data_enabled,
            }
            # Only send the wiki field when a local description exists;
            # omitting it leaves the bio on AI Arena unchanged.
            if (description := get_bot_description()) is not None:
                request_data["wiki_article_content"] = description
            request_files = {
                "bot_zip": bot_zip,
            }
            logger.info(URL)
            response = requests.patch(
                URL, headers=request_headers, data=request_data, files=request_files
            )
            logger.info(response)
            logger.info(response.content)
