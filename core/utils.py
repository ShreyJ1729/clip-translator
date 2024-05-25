import pathlib
import core.config as config


def download_video(youtube_video_id: pathlib.Path):
    """
    Given a youtube video id, downloads the video to cache and returns the path.
    """

    import subprocess

    # create cache directories if they don't exist
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_file = config.VIDEO_DIR / f"{youtube_video_id}.mp4"

    if not video_file.exists():
        try:
            print(f"Downloading video {youtube_video_id}")
            subprocess.run(
                [
                    "yt-dlp",
                    "--format",
                    "mp4",
                    "--quiet",
                    "--output",
                    str(video_file),
                    f"https://www.youtube.com/watch?v={youtube_video_id}",
                ]
            )
        except Exception as e:
            print(e)
            raise Exception("Error downloading video.")

    print(f"Downloaded video to {video_file}")
    return video_file


def extract_audio(video_file: pathlib.Path):
    """
    Given a path to a video file, extracts audio to cache.
    """
    import ffmpeg

    # create cache directories if they don't exist
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # download video to shared volume
    video_id = video_file.stem
    audio_file = config.AUDIO_DIR / f"{video_id}.mp3"

    if not audio_file.exists():
        print(f"Extracting audio from video to {audio_file}")
        ffmpeg.input(str(video_file)).output(
            str(audio_file),
            ac=1,
            ar=16000,
        ).run()

    return audio_file
