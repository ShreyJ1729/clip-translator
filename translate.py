import config
from config import stub, app_image, volume, mounts
import pathlib
import json
from typing import Dict, Any

def translate_text(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
    """
    Given a text, translates it to target language using gpt=3.5-turbo.
    """

    from openai import OpenAI
    import os
    import time
    import dotenv
    dotenv.load_dotenv("/root/.env")

    client = OpenAI()
    print("Sending request to OpenAI API for translation...")
    t0 = time.time()
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

    print(f"Translated text ({len(text)} characters; {len(text.split())} words) from {source_language} to {target_language} in {time.time() - t0:.2f} seconds.")

    return translated_text