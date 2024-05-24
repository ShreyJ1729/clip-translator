# cuda-shell.py
from modal import App, Image

image = Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
app = App(image=image)


@app.function(gpu="T4")
def f():
    import subprocess

    subprocess.run(["nvidia-smi"])
