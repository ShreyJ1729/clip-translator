import modal

wav2lip_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .run_commands(
        "git clone https://github.com/ShreyJ1729/Wav2Lip.git /root/Wav2Lip",
        "cd /root/Wav2Lip && pip install -r requirements.txt",
    )
)
