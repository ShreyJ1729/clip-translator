"""
Given a dubbed video of a speaker (in frame) whose lips are not in sync with the audio, this code uses Video-Retalking to realistically lip-sync.

Sequential model pipeline:
- Just apply Video-Retalking

Notes to remember:
- better logging and error handling
- add upscale for lip-synced video to increase visual quality of lips
- create feature to email one-time use download link for processed videos with post-download cleanup (delete)
= benchmark cpu/memory/time for each function to see how to best split containers
"""

import modal
import config
import pathlib
from config import app, app_image, volume, mounts
import lipsync


@app.function(
    image=app_image,
    mounts=mounts,
    network_file_systems={config.CACHE_DIR: volume},
    timeout=600,
)
@modal.web_endpoint(method="GET")
def translate_video(youtube_video_id: str):
    """
    Given a youtube video id of a dubbed video, lip-syncs and returns it.
    """

    from fastapi.responses import FileResponse

    download_video_extract_audio(youtube_video_id)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LIPSYNCED_DIR.mkdir(parents=True, exist_ok=True)

    # identify paths
    video_file = config.VIDEO_DIR / f"{youtube_video_id}.mp4"
    audio_file = config.AUDIO_DIR / f"{youtube_video_id}.mp3"

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


def download_video_extract_audio(youtube_video_id: str):
    """
    Given a youtube video id, downloads the video and extracts audio to cache.
    """
    import ffmpeg
    import subprocess

    # create cache directories if they don't exist
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_file = config.VIDEO_DIR / f"{youtube_video_id}.mp4"
    audio_file = config.AUDIO_DIR / f"{youtube_video_id}.mp3"

    if not video_file.exists():
        try:
            print(f"Downloading video {youtube_video_id}")
            subprocess.run(
                [
                    "yt-dlp",
                    # "--quiet",
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

    # extract audio from video
    if not audio_file.exists():
        print(f"Extracting audio from video {youtube_video_id}")
        ffmpeg.input(str(video_file)).output(
            str(audio_file),
            ac=1,
            ar=16000,
        ).run()
