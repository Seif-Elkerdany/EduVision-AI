# Smart Classroom Speech Module

This module provides a complete end-to-end speech recognition pipeline designed for a Smart Classroom environment. It uses **OpenAI's Whisper API** for highly accurate transcription, combined with a custom question detection engine to parse student queries and statements.

## Features

- **Push-to-Talk Recording**: Uses `sounddevice` to capture audio until a silence threshold is reached.
- **Advanced Noise Reduction**: Implements a multi-pass noise reduction pipeline using `noisereduce` and `librosa` to clean up classroom background noise (AC hum, fans, etc.) before transcription.
- **OpenAI Whisper Transcription**: Fast and highly accurate speech-to-text using Whisper's `base` model.
- **Intelligent Sentence Splitting**: Smartly divides transcribed blocks of text into logical sentences based on discourse markers, even when the transcription lacks punctuation.
- **Deep Question Detection**: Uses a 5-strategy detector (Explicit, Opener, Embedded, Tag, and Indirect) to identify whether a sentence is a question or a statement, even if the question is buried mid-sentence.
- **Intonation Heuristics**: Detects rising pitch at the end of the audio as a hint for a question.
- **Data Logging**: Automatically structures the text into clauses with labels (`Question` / `Statement`) and saves the session log into `nlp_log.json`.

## Prerequisites

Make sure you have installed the required dependencies:

```bash
pip install openai-whisper noisereduce librosa soundfile sounddevice speechrecognition numpy scipy
```

*Note: You may also need to install `ffmpeg` on your system for Whisper to process audio correctly.*

## Usage

Run the script from your terminal:

```bash
python Speech_recognition.py
```

### Workflow:
1. **Start**: Press `ENTER` to begin recording your voice.
2. **Speak**: Speak into your microphone. The recording automatically stops after 2 seconds of silence or a maximum of 60 seconds.
3. **Processing**: The module will automatically clean the audio, run transcription, and analyze the text.
4. **Output**: The console will display the raw text and the categorized sentences (Question vs. Statement).
5. **Logs**: The structured data is appended to `nlp_log.json`, while audio files are saved locally as `raw_audio.wav` and `clean_audio.wav`.

## Output Files

- `raw_audio.wav`: The original unedited audio captured from the microphone.
- `clean_audio.wav`: The audio after stationary noise reduction and pre-emphasis filtering.
- `nlp_log.json`: The JSON database storing the recognized sentences, their question/statement classification, and timestamps.
