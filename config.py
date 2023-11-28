import modal
import dataclasses
import logging
import pathlib


stub = modal.Stub("single-speaker-clip-translator")

mounts = [
    modal.Mount.from_local_file(".env", remote_path="/root/.env"),
    modal.Mount.from_local_file("requirements.txt", remote_path="/root/requirements.txt"),
    ]

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install_from_requirements("requirements.txt")
)

lipsync_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .run_commands(
        "git clone https://github.com/ShreyJ1729/Wav2Lip.git /root/Wav2Lip",
        "cd /root/Wav2Lip && pip install -r requirements.txt")
    )

volume = modal.NetworkFileSystem.persisted("cliptranslator-model-weights-cache")

@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# root directory for all cached data
CACHE_DIR = "/cache"

# model checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

# downloaded video files
VIDEO_DIR = pathlib.Path(CACHE_DIR, "video")

# audio files extracted from video
AUDIO_DIR = pathlib.Path(CACHE_DIR, "audio")

# transcript json files for each audio file
TRANSCRIPT_DIR = pathlib.Path(CACHE_DIR, "transcript")

# processed (lip-synced) video files
LIPSYNCED_DIR = pathlib.Path(CACHE_DIR, "lipsynced")

supported_whisper_models = {
    "tiny": ModelSpec(name="tiny", params="39M", relative_speed=32),
    # Takes around 3-10 minutes to transcribe a podcast, depending on length.
    "base": ModelSpec(name="base", params="74M", relative_speed=16),
    "small": ModelSpec(name="small", params="244M", relative_speed=6),
    "medium": ModelSpec(name="medium", params="769M", relative_speed=2),
    # Very slow. Will take around 45 mins to 1.5 hours to transcribe.
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}


DEFAULT_MODEL = supported_whisper_models["base"]