import modal
import logging
import pathlib

app = modal.App("dubbsync")

mounts = [
    # modal.Mount.from_local_file(".env", remote_path="/root/.env"),
    modal.Mount.from_local_file(
        "requirements.txt", remote_path="/root/requirements.txt"
    ),
]

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install_from_requirements("requirements.txt")
)


lipsync_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.9")
    .apt_install(
        "ffmpeg",
        "git",
        "cmake",
        "build-essential",
        "libopenblas-dev",
        "liblapack-dev",
        "pkg-config",
        "wget",
        "software-properties-common",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get clean",
        "apt-get -y install cuda",
    )
    .run_commands(
        "export PATH=~/usr/bin/cmake:$PATH && export CUDA_HOME=/opt/conda && export PATH=$PATH:$CUDA_HOME/bin && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 && export CPATH=$CUDA_HOME/include:$CPATH && export CUDNN_INCLUDE_DIR=$CUDA_HOME/include && export CUDNN_LIB_DIR=$CUDA_HOME/lib64",
        "pip install --upgrade setuptools pip",
        "pip3 install torch torchvision torchaudio",
        # build dlib from source with cuda support
        "git clone https://github.com/davisking/dlib.git && cd dlib && mkdir build && cd build && cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DCUDA_TOOLKIT_ROOT_DIR=/opt/conda && cmake --build . && cd .. && python setup.py install --set DLIB_USE_CUDA=1",
        "pip install git+https://github.com/XPixelGroup/BasicSR kornia==0.5.1 face-alignment==1.3.4 ninja==1.10.2.3 einops==0.4.1 facexlib==0.2.5 librosa==0.9.2 numpy==1.20.0",
        "python -c 'import dlib; print(dlib.__version__); print(dlib.DLIB_USE_CUDA); print(dlib.cuda.get_num_devices())'",
    )
    .run_commands(
        "git clone https://github.com/OpenTalker/video-retalking.git /root/video-retalking"
    )
)

volume = modal.NetworkFileSystem.from_name("dubbsync-cache", create_if_missing=True)


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# root directory for all cached data (mount point for nfs)
CACHE_DIR = "/cache"

# model checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

# downloaded video files
VIDEO_DIR = pathlib.Path(CACHE_DIR, "video")

# audio files extracted from video
AUDIO_DIR = pathlib.Path(CACHE_DIR, "audio")

# processed (lip-synced) video files
LIPSYNCED_DIR = pathlib.Path(CACHE_DIR, "lipsynced")
