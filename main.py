"""
Given a dubbed video of a speaker (in frame) whose lips are not in sync with the audio, this code uses Video-Retalking to realistically lip-sync.

Multi-speaker pipeline:
- Eleven Labs dubbing API
- Diarization to determine num speakers and timestamps
- Talking head detection to crop face of active speaker
- Video-Retalking to lip-sync audio to cropped face
"""

import modal
import config
from config import app, app_image, cache, mounts
import os
import pathlib
import time
import lipsync
import requests
import dubbing_api


@app.function(
    image=app_image,
    mounts=mounts,
    volumes={config.CACHE_DIR: cache},
    timeout=60 * 60 * 1,
)
@modal.web_endpoint(method="GET")
def translate_video(youtube_video_id: str):
    """
    Given a youtube video id of a video, translates, lip-syncs and returns it.
    """

    from fastapi.responses import FileResponse

    video_file = download_video(youtube_video_id)

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

    # audio_file = extract_audio(dubbed_file)

    audio_file = extract_audio(video_file)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LIPSYNCED_DIR.mkdir(parents=True, exist_ok=True)

    # lip-sync video to new audio
    lipsynced_file = config.LIPSYNCED_DIR / f"{youtube_video_id}.mp4"
    lipsync.perform_lip_sync.remote(video_file, audio_file, lipsynced_file)

    # cleanup, deleting cached audio and video files
    audio_file.unlink()
    video_file.unlink()
    print("Deleted cached audio and video files.")

    # return as Fastapi File Response
    return FileResponse(
        str(lipsynced_file),
        media_type="video/mp4",
        filename=f"{youtube_video_id}-dubbsynced.mp4",
    )


def download_video(youtube_video_id: pathlib.Path):
    """
    Given a youtube video id, downloads the video to cache and returns the path.
    """

    import subprocess

    # create cache directories if they don't exist
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_file = config.VIDEO_DIR / f"{youtube_video_id}.mp4"

    if not video_file.exists():
        try:
            print(f"Downloading video {youtube_video_id}")
            subprocess.run(
                [
                    "yt-dlp",
                    "--format",
                    "mp4",
                    "--output",
                    str(video_file),
                    f"https://www.youtube.com/watch?v={youtube_video_id}",
                ]
            )
        except Exception as e:
            print(e)
            raise Exception("Error downloading video.")

    return video_file


def extract_audio(video_file: pathlib.Path):
    """
    Given a path to a video file, extracts audio to cache.
    """
    import ffmpeg
    import subprocess

    # create cache directories if they don't exist
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_id = video_file.stem
    audio_file = config.AUDIO_DIR / f"{video_id}.mp3"

    if not audio_file.exists():
        print(f"Extracting audio from video {audio_file}")
        ffmpeg.input(str(video_file)).output(
            str(audio_file),
            ac=1,
            ar=16000,
        ).run()

    return audio_file
