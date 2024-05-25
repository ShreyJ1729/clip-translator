import modal
import logging
import pathlib
import modal.gpu

app = modal.App("dubbsync")

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install("ffmpeg-python", "yt-dlp", "python-dotenv")
)

mounts = [
    modal.Mount.from_local_dir("media", remote_path="/root/media"),
]

# contains all model files for lip-syncing
cache = modal.Volume.from_name("cliptranslator-cache", create_if_missing=True)

# root directory for all cached data (mount point for nfs)
CACHE_DIR = "/cache"

# model checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

# downloaded video files
VIDEO_DIR = pathlib.Path(CACHE_DIR, "video")

# dubbed video files
DUBBED_DIR = pathlib.Path(CACHE_DIR, "dubbed")

# audio files extracted from video
AUDIO_DIR = pathlib.Path(CACHE_DIR, "audio")

# processed (lip-synced) video files
LIPSYNCED_DIR = pathlib.Path(CACHE_DIR, "lipsynced")
