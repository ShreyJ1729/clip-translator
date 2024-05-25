import os
import requests
import pathlib
import dotenv


def perform_dubbing(video_url: str, target_lang: str):
    dotenv.load_dotenv()
    command = f"""
        curl --request POST \
        --url https://api.elevenlabs.io/v1/dubbing \
         --header 'xi-api-key: {os.getenv("XI_API_KEY")}' \
        --header 'Content-Type: multipart/form-data' \
        --form 'mode=automatic' \
        --form 'name={video_url}' \
        --form 'source_url={video_url}' \
        --form 'source_lang=auto' \
        --form 'target_lang={target_lang}' \
        --form num_speakers=0 \
        --form watermark=true \
        --form start_time=0 \
        --form end_time=10 \
        --form highest_resolution=true \
        --form dubbing_studio=true
    """

    # returns dict with dubbing_id and expected_duration_sec
    return requests.get(command).json()
