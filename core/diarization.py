import os
import dotenv
import time
from pyannote.audio.pipelines.utils.hook import ProgressHook

dotenv.load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
from pyannote.audio import Pipeline

# Initialize the diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=HUGGINGFACE_API_KEY)

start = time.time()

with ProgressHook() as hook:
    diarization = pipeline("audio2.wav", hook=hook, num_speakers=2)


print(f"Processing took {time.time() - start:.1f}s.")

# Print the diarization result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker} speaks from {turn.start} to {turn.end}")
