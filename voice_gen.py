import pathlib
from typing import Dict

def add_voice(voice_name: str, audio_file: pathlib.Path, voice_description: str, voice_labels: Dict[str, str]):
    """
    Given a voice name, audio path, voice description, and voice labels, uploads the data to 11labs to train voice.
    Returns the voice id.
    """
    import requests
    import os
    import dotenv
    import json
    from requests.structures import CaseInsensitiveDict
    from elevenlabs import set_api_key, clone

    dotenv.load_dotenv("/root/.env")
    set_api_key(os.getenv("ELEVENLABS_API_KEY"))

    # clone voice and return voice object
    voice = clone(
        name=voice_name,
        description=voice_description,
        labels=voice_labels,
        files=[str(audio_file)],
    )

    return voice

def generate(translated_text: str, voice: str, target_language: str):
    """
    Given translated text, voice id, and target language, generates a voice.
    """
    import requests
    import os
    import dotenv
    import json
    from requests.structures import CaseInsensitiveDict
    from elevenlabs import set_api_key, generate

    dotenv.load_dotenv("/root/.env")
    set_api_key(os.getenv("ELEVENLABS_API_KEY"))

    # get raw audio bytes generated from elevenlabs
    audio = generate(text=translated_text[:250], voice=voice)

    return audio