import sys
import modal
import pathlib
from config import lipsync_image, stub, volume, mounts
import config
import time

@stub.function(image=lipsync_image, mounts=mounts, network_file_systems={config.CACHE_DIR: volume}, gpu="any", concurrency_limit=1)
def perform_lip_sync(video_file: pathlib.Path, audio_file: pathlib.Path, output_file: pathlib.Path):
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
    subprocess.run(f"cp {s3fd_path} Wav2Lip/face_detection/detection/sfd/s3fd.pth", shell=True)

    # print the gpu in use
    gpu_in_use = subprocess.run("nvidia-smi", shell=True, capture_output=True).stdout.decode("utf-8")
    print(f"GPU in use: {gpu_in_use}")

    # perform lip sync
    print(f"Performing lip sync combining {str(video_file)} and {str(audio_file)}, saving to {str(output_file)}...")
    t0 = time.time()
    
    command = f"cd /root/Wav2Lip && python inference.py --checkpoint_path {str(wave2lip_gan_path)} --face {str(video_file)} --audio {str(audio_file)} --outfile {str(output_file)}"
    process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE)
    counter = 0
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.write(c)
        counter += 1
        if counter % 100 == 0:
            sys.stdout.flush()

    print(f"Lip-synced in {time.time() - t0:.2f} seconds. Saved to {output_file}.")