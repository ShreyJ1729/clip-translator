import sys
import modal
import pathlib
import core.config as config
from config import app, volume, mounts
from core.images.wav2lip import wav2lip_image
import time


@app.function(
    image=wav2lip_image,
    mounts=mounts,
    volumes={config.CACHE_DIR: volume},
    gpu="A10G",
    cpu=4,
    memory=1024 * 16,
    timeout=60 * 60 * 1,
)
def perform_lip_sync(
    video_file: pathlib.Path, audio_file: pathlib.Path, output_file: pathlib.Path
):
    """
    Given a video and audio file, performs lip sync using wav2lip and saves to output file.
    """
    import subprocess

    # download model weights to corresponding directories if not already present
    # wave2lip gan model weights
    wave2lip_gan_path = config.MODEL_DIR / "wav2lip_gan.pth"
    if not wave2lip_gan_path.exists():
        raise Exception("wav2lip_gan.pth not found in model directory.")

    # s3fd face detection model weights
    s3fd_path = config.MODEL_DIR / "s3fd.pth"
    if not s3fd_path.exists():
        raise Exception("s3fd.pth not found in model directory.")

    # copy s3fd model weights to wav2lip directory
    subprocess.run(
        f"cp {s3fd_path} Wav2Lip/face_detection/detection/sfd/s3fd.pth", shell=True
    )

    # perform lip sync
    print(
        f"Performing lip sync combining {str(video_file)} and {str(audio_file)}, saving to {str(output_file)}..."
    )
    t0 = time.time()

    command = f"cd /root/Wav2Lip && python inference.py --checkpoint_path {str(wave2lip_gan_path)} --face {str(video_file)} --audio {str(audio_file)} --outfile {str(output_file)}"
    process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE)

    # stream output line by line to stdout
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)

    # wait for process to finish and check for errors
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Lip-synced in {time.time() - t0:.2f} seconds. Saved to {output_file}.")
