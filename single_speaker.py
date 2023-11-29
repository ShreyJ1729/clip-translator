"""
This code translates a video of a single-speaker to another language while maintaining the original speaker\'s vocal identity and lip-syncing to the new audio.

Sequential model pipeline:

- Send audio to 11labs to train voice
- Transcribe input video with Whisper STT
- Use gpt4-turbo on transcription to translate to target language
- Send translated text to 11labs TTS to generate new audio
- Lip-sync person in video to audio using Wav2Lip


TODO:
- match up speaking pace of translated audio to original video through cutting at pauses, generating audio for each segment, and speeding up/slowing down audio in each segment to match the cut, then restitching
- Replace 11labs with https://github.com/yl4579/StyleTTS2
- add model to clean up audio, removing music/background noise from voices before sending to elevenlabs, and re-adds it after receiving generated audio
- add model for accurate voice quality/labels generation to send to elevenlabs
- better logging and error handling
- add upscale for lip-synced video to increase visual quality of lips
- try using stable diffusion model for lip-syncing since wav2lip doesn't work well (https://github.com/OpenTalker/video-retalking)
- create feature to email one-time use download link for processed videos with post-download cleanup (delete)
= benchmark cpu/memory/time for each function to see how to best split containers
"""

import modal
import config
import pathlib
from config import stub, app_image, volume, mounts
from transcribe import transcribe_audio
from translate import translate_text
import voice_gen
import lipsync

@stub.function(image=app_image, mounts=mounts, network_file_systems={config.CACHE_DIR: volume}, timeout=600)
@modal.web_endpoint(method="GET")
def translate_video():
    """
    Given a youtube video id, translates it to target language.
    """

    from fastapi.responses import FileResponse


    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LIPSYNCED_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


    voice_name = "voice 1"
    voice_description = ""
    voice_labels = {}

    video_file = config.VIDEO_DIR / f"betterhelpgirl.mp4"
    audio_file = config.AUDIO_DIR / f"betterhelpgirl.wav"


    # lip-sync video to new audio
    lipsynced_file = config.LIPSYNCED_DIR / f"betterhelpgirl.mp4"
    lipsync.perform_lip_sync.remote(video_file, audio_file, lipsynced_file)

    return FileResponse(lipsynced_file, media_type="video/mp4")


def download_video_extract_audio(youtube_video_id: str):
    """
    Given a youtube video id, downloads the video and extracts audio.
    """
    import ffmpeg
    import subprocess

    # create cache directories if they don't exist
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_file = config.VIDEO_DIR / f"{youtube_video_id}.mp4"
    audio_file = config.AUDIO_DIR / f"{youtube_video_id}.mp3"

    if not video_file.exists():
        try:
            print(f"Downloading video {youtube_video_id}")
            subprocess.run(
                [
                    "yt-dlp",
                    # "--quiet",
                    "--format",
                    "mp4",
                    "--output",
                    str(video_file),
                    f"https://www.youtube.com/watch?v={youtube_video_id}",
                ]
            )
        except Exception as e:
            print(e)
            raise Exception("Error downloading video.")
    
    # extract audio from video
    if not audio_file.exists():
        print(f"Extracting audio from video {youtube_video_id}")
        ffmpeg.input(str(video_file)).output(
            str(audio_file), ac=1, ar=16000,
        ).run()
