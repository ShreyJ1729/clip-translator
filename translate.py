import config
from config import stub, app_image, volume, mounts
import pathlib
import json
from typing import *

@stub.function(image=app_image, mounts=mounts, network_file_systems={config.CACHE_DIR: volume}, timeout=900) 
def translate_text(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
    """
    Given a text, translates it to target language using gpt4-turbo.
    """

    from openai import OpenAI
    import os

    import dotenv
    dotenv.load_dotenv("/root/.env")

    client = OpenAI()

    chatcompletion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text into {target_language}. Do not respond with anything other than the translated text. \n\n{text}",
            }
        ],
        model="gpt-3.5-turbo",
    )

    translated_text = chatcompletion.choices[0].message.content

    return translated_text