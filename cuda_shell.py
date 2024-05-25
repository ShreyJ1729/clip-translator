import modal
from config import app, volume, lipsync_image, CACHE_DIR


@app.function(
    image=lipsync_image,
    network_file_systems={CACHE_DIR: volume},
    gpu=modal.gpu.T4(),
)
def main():
    print("Hello, world!")
