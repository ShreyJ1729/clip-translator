import sys
import pathlib
from config import lipsync_image, app, volume, mounts
import config
import time


@app.function(
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
                    export CUDA_HOME=/opt/conda && 
                    export PATH=$PATH:$CUDA_HOME/bin &&
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 &&
                    export CPATH=$CUDA_HOME/include:$CPATH &&
                    export CUDNN_INCLUDE_DIR=$CUDA_HOME/include &&
                    export CUDNN_LIB_DIR=$CUDA_HOME/lib64 &&

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
