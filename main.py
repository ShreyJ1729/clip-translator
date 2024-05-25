"""
Given a dubbed video of a speaker (in frame) whose lips are not in sync with the audio, this code uses Video-Retalking to realistically lip-sync.

Multi-speaker pipeline:
- Eleven Labs dubbing API
- Diarization to determine num speakers and timestamps
- Talking head detection to crop face of active speaker
- Video-Retalking to lip-sync audio to cropped face
"""

import modal
import core.utils as utils
import core.config as config
from core.config import app, app_image, cache, mounts
import os
import pathlib
import time
import core.lipsync as lipsync
import requests
import core.dubbing_api as dubbing_api


@app.function(
    image=app_image,
    mounts=mounts,
    volumes={config.CACHE_DIR: cache},
    timeout=60 * 60 * 1,
)
@modal.web_endpoint(method="GET")
def process_video(youtube_video_id: str):
    """
    Given a youtube video id of a video, translates, lip-syncs and returns it.
    """

    from fastapi.responses import FileResponse

    video_file = utils.download_video(youtube_video_id)

    #     api_result = dubbing_api.perform_dubbing(youtube_video_id, "spanish")
    #     dubbing_id = api_result["dubbing_id"]
    #     time.sleep(api_result["expected_duration_sec"])
    # while True:
    #     get_metadata_command = f"""curl --request GET \
    #         --url https://api.elevenlabs.io/v1/dubbing/{api_result['dubbing_id']} \
    #         --header 'xi-api-key: {os.getenv("XI_API_KEY")}'"""
    #     metadata = requests.get(get_metadata_command).json()
    #     if metadata["status"] == "completed":
    #         break
    #     time.sleep(1)

    # language_code = "es"
    # get_dubbed_video_command =f"""curl --request GET \
    #         --url https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{language_code} \
    #         --header 'xi-api-key: {os.getenv("XI_API_KEY")}'"""
    # dubbed_video = requests.get(get_dubbed_video_command)

    # dubbed_file = config.DUBBED_DIR / f"{youtube_video_id}.mp4"
    # with open(dubbed_file, "wb") as f:
    #     f.write(dubbed_video.content)

    # audio_file = utils.extract_audio(dubbed_file)

    audio_file = utils.extract_audio(video_file)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LIPSYNCED_DIR.mkdir(parents=True, exist_ok=True)

    # lip-sync video to new audio
    lipsynced_file = config.LIPSYNCED_DIR / f"{youtube_video_id}.mp4"
    lipsync.perform_lip_sync.remote(video_file, audio_file, lipsynced_file)

    # cleanup, deleting cached audio and video files
    audio_file.unlink()
    video_file.unlink()
    print("Deleted cached audio and video files.")
