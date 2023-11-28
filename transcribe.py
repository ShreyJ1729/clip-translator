import config
from config import stub, app_image, volume, mounts
import pathlib
import json
from typing import Dict, Iterator, Tuple

def transcribe_audio(
    audio_file: pathlib.Path,
    model: config.ModelSpec = config.DEFAULT_MODEL,
):
    import whisper
    import time

    # pre-download whisper model to shared volume for parallelization since _download is not thread-safe
    print(f"Downloading whisper model '{model.name}' or validating checksum")
    whisper._download(whisper._MODELS[model.name], config.MODEL_DIR, False)

    # split audio into segments for parallel processing based on silence
    segment_gen = split_silences(str(audio_file))

    # transcribe each segment in parallel
    t0 = time.time()
    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_file=audio_file, model=model)
    ):
        output_text += result["text"]
        output_segments += result["segments"]
    
    print(f"Transcribed segments in {time.time() - t0:.2f} seconds.")

    result = {
        "text": output_text,
        "segments": output_segments,
    }

    # save transcript to file
    transcript_file = config.TRANSCRIPT_DIR / f"{audio_file.stem}.json"
    with open(transcript_file, "w") as f:
        json.dump(result, f)
    
    return result
 


@stub.function(
    image=app_image,
    network_file_systems={config.CACHE_DIR: volume},
    cpu=4,
)
def transcribe_segment(
    start: float,
    end: float,
    audio_file: pathlib.Path,
    model: config.ModelSpec,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(audio_file))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=config.MODEL_DIR
        )
        result = model.transcribe(f.name, fp16=use_gpu)

    print(
        f"- Transcribed segment {start:.2f} to {end:.2f} ({end - start:.2f}s duration) in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result



def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re

    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    print(f"Split {path} into {num_segments} segments")