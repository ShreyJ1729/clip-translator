## ClipTranslator

Single-speaker model pipeline:

- Send audio to 11labs for voice training
- Whisper STT
- gpt3.5-turbo for translation
- 11labs TTS
- Wav2Lip

Multi-speaker Model pipeline:

- Speaker Diarization (huggingface pyannote/speaker-diarization-3.0)
- Whisper STT
- gpt3.5-turbo for translation
- 11labs TTS
- Active speaker detection (https://github.com/sra2/spell)
- Wav2Lip on active speaker

- Identify speakers in video and annotate segments based on who's talking
- Extract audio from video segments
- - Train 11labs voice model for each speaker
- - Convert audio to text using Whisper STT + translate to given language
- Convert text to audio using 11labs TTS
- Use Wav2Lip to generate lip-synced video for each speaker
- Stitch together all video/audio segments into final video
