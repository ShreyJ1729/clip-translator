"""
This code translates a video of a single-speaker to another language while maintaining the original speaker\'s vocal identity and lip-syncing to the new audio.

Sequential model pipeline:

- Transcribe input video with Whisper STT
- Use gpt4-turbo on transcription to translate to target language
- Send translated text to 11labs TTS to generate new audio
- Lip-sync person in video to audio using Wav2Lip

"""

import modal
import json
import pathlib
import config
from config import stub, app_image, volume, mounts
from typing import *
from transcribe import transcribe_audio
from translate import translate_text

@stub.function(image=app_image, mounts=mounts, network_file_systems={config.CACHE_DIR: volume}, timeout=900)
@modal.web_endpoint(method="GET")
def translate_video(youtube_video_id: str, target_language: str = "french"):
    """
    Given a youtube video id, translates it to target language.
    """
    download_video_extract_audio.local(youtube_video_id)

    # identify paths
    video_path = config.VIDEO_DIR / f"{youtube_video_id}.mp4"
    audio_path = config.AUDIO_DIR / f"{youtube_video_id}.mp3"
    transcription_path = config.TRANSCRIPT_DIR / f"{youtube_video_id}.json"

    # send audio data to elevenlabs for async voice training while we transcribe/translate
    # TODO
    
    # transcribe and translate audio to target language
    original_transcription = transcribe_audio.local(audio_path, transcription_path)
    translated_transcription = translate_text.local(original_transcription["text"], target_language)

    # once elevenlabs voice finished training, send translated text to elevenlabs
    # TODO

    # lip-sync video to new audio
    # TODO

    # save lip-synced video to shared volume and generate download link
    # TODO

    # return download link
    # TODO
    return "download_link"

@stub.function(image=app_image, mounts=mounts, network_file_systems={config.CACHE_DIR: volume}, timeout=900)
def download_video_extract_audio(youtube_video_id: str):
    """
    Given a youtube video id, downloads the video and extracts audio.
    """
    import ffmpeg
    import subprocess

    # create cache directories if they don't exist
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_path = config.VIDEO_DIR / f"{youtube_video_id}.mp4"
    audio_path = config.AUDIO_DIR / f"{youtube_video_id}.mp3"

    if not video_path.exists():
        print(f"Downloading video {youtube_video_id}")
        subprocess.run(
            [
                "yt-dlp",
                # "--quiet",
                "--format",
                "mp4",
                "--output",
                str(video_path),
                f"https://www.youtube.com/watch?v={youtube_video_id}",
            ]
        )
    
    # extract audio from video
    if not audio_path.exists():
        print(f"Extracting audio from video {youtube_video_id}")
        ffmpeg.input(str(video_path)).output(
            str(audio_path), ac=1, ar=16000,
        ).run()
