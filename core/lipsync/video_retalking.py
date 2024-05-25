import time
import sys
import pathlib
import core.config as config
from core.config import app, cache
from core.images.video_retalking import video_retalking_image


@app.function(
    image=video_retalking_image,
    volumes={config.CACHE_DIR: cache},
    gpu="A10G",
    memory=1024 * 16,
    timeout=60 * 60 * 1,
)
def lipsync_video_retalking(
    video_file: pathlib.Path, audio_file: pathlib.Path, output_file: pathlib.Path
):
    """
    Given a video and audio file, performs lip sync using video-retalking and saves to output file.
    """
    import subprocess

    # copy model files from cache to container for faster access
    checkpoints_dest = pathlib.Path("/root/video-retalking/checkpoints")
    if not (checkpoints_dest).exists():
        print(f"Copying model files from {config.MODEL_DIR} to {checkpoints_dest}...")
        subprocess.run(
            f"rsync -ah --progress {config.MODEL_DIR}/* {checkpoints_dest}",
            shell=True,
            check=True,
        )

    print(
        f"Performing lip sync combining {str(video_file)} and {str(audio_file)}, saving to {str(output_file)}..."
    )
    t0 = time.time()

    command = f"""
                    export CUDA_HOME=/usr/local/cuda && 
                    export PATH=$PATH:$CUDA_HOME/bin &&
                    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64 &&
                    export CPATH=$CUDA_HOME/include:$CPATH &&
                    export CUDNN_INCLUDE_DIR=$CUDA_HOME/include &&
                    export CUDNN_LIB_DIR=$CUDA_HOME/lib64 &&

                    cd /root/video-retalking &&
                    python inference.py --checkpoints {checkpoints_dest} --face {str(video_file)} --audio {str(audio_file)} --outfile {str(output_file)}"""

    process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE)

    # stream output line by line to stdout
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)

    # wait for process to finish and check for errors
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Lip-synced in {time.time() - t0:.2f} seconds. Saved to {output_file}.")
