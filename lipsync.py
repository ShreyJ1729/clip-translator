import sys
import modal
import pathlib
from config import lipsync_image, stub, volume, mounts
import config
import time

import subprocess
import os


def check_cuda_installation():
    # List of common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.1",
        "/usr/local/cuda-11",
        "/opt/cuda",
        "/opt/cuda-11.1",
        "/opt/cuda-11",
    ]

    print("Searching for CUDA Toolkit installation...")

    # Check each path for nvcc
    for path in cuda_paths:
        nvcc_path = os.path.join(path, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            print(f"CUDA Toolkit found at: {path}")
            return

    # Check if nvcc is in the system PATH
    try:
        nvcc_location = subprocess.check_output(["which", "nvcc"], text=True).strip()
        if nvcc_location:
            cuda_path = os.path.dirname(os.path.dirname(nvcc_location))
            print(f"CUDA Toolkit found at: {cuda_path}")
            return
    except subprocess.CalledProcessError:
        pass

    # Final message if CUDA Toolkit is not found
    print("CUDA Toolkit not found in common locations. Please check your installation.")


@stub.function(
    image=lipsync_image,
    mounts=mounts,
    network_file_systems={config.CACHE_DIR: volume},
    gpu="any",
    cpu=2,
    memory=8192,
    concurrency_limit=1,
    timeout=600,
)
def perform_lip_sync(
    video_file: pathlib.Path, audio_file: pathlib.Path, output_file: pathlib.Path
):
    """
    Given a video and audio file, performs lip sync using video-retalking and saves to output file.
    """
    import subprocess

    # copy model files from cache to container
    print(
        subprocess.check_output("locate cuda | grep /cuda$", shell=True).decode("utf-8")
    )

    local_destination = pathlib.Path("/root/video-retalking/")
    if not (local_destination / "checkpoints").exists():
        print(f"Copying model files from {config.MODEL_DIR} to {local_destination}...")
        subprocess.run(
            f"cp -r {config.MODEL_DIR} {local_destination}", shell=True, check=True
        )

    # print the gpu in use
    gpu_in_use = subprocess.run(
        "nvidia-smi", shell=True, capture_output=True
    ).stdout.decode("utf-8")
    print(gpu_in_use)

    # perform lip sync
    print(
        f"Performing lip sync combining {str(video_file)} and {str(audio_file)}, saving to {str(output_file)}..."
    )
    t0 = time.time()

    command = f"""
                    export CUDA_HOME=/usr/bin/ && 
                    export PATH=$PATH:$CUDA_HOME/bin &&
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 &&
                    cd /root/video-retalking &&
                    python inference.py --checkpoint_path --face {str(video_file)} --audio {str(audio_file)} --outfile {str(output_file)}"""

    process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE)

    # stream output line by line to stdout
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)

    # wait for process to finish and check for errors
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Lip-synced in {time.time() - t0:.2f} seconds. Saved to {output_file}.")
