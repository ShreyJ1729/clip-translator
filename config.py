import modal
import logging
import pathlib

stub = modal.Stub("orpheus")

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
    modal.Image.micromamba(python_version="3.9")
    .apt_install("ffmpeg", "git", "cmake", "build-essential")
    .run_commands("export PATH=~/usr/bin/cmake:$PATH")
    .micromamba_install( ["cudatoolkit=11.1", "cudnn", "cuda-nvcc", ], channels=["conda-forge", "nvidia"], gpu="any")
    .run_commands(
        "pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
        "pip install basicsr==1.4.2 kornia==0.5.1 face-alignment==1.3.4 ninja==1.10.2.3 einops==0.4.1 facexlib==0.2.5 librosa==0.9.2 dlib==19.24.0 numpy==1.20.0")
    .run_commands("git clone https://github.com/OpenTalker/video-retalking.git /root/video-retalking"))

volume = modal.NetworkFileSystem.persisted("orpheus-cache")

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

# processed (lip-synced) video files
LIPSYNCED_DIR = pathlib.Path(CACHE_DIR, "lipsynced")
