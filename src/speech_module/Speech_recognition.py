"""
Smart Classroom Speech Module — Google Speech API Edition
=========================================================
Uses sr.recognize_google() exactly like your original code,
with all accuracy improvements applied on top:
  • Multi-pass noise reduction
  • Per-sentence question detection (catches questions mid-conversation)
  • Intonation heuristic
  • Better recording settings

Dependencies:
    pip install noisereduce librosa soundfile speechrecognition numpy scipy
"""

import whisper
import noisereduce as nr
import numpy as np
import json
import re
import os
import time
from datetime import datetime

# Whisper model configuration.
# Use a smaller model for faster runtime: tiny, base, small, medium, large.
WHISPER_MODEL_NAME = "tiny"
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"⏳ Loading Whisper model '{WHISPER_MODEL_NAME}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return _whisper_model



# ── 1. QUESTION DETECTION ENGINE ─────────────────────────────────────────────
#
# Strategy: 5 independent detectors scan ANYWHERE in the sentence.
# Any single hit → question. This catches questions buried in the middle
# of longer speech like "I was thinking, but what do you mean by that?"
#
# Detector 1 — Explicit question mark
# Detector 2 — Classic question word at START of sentence
# Detector 3 — Embedded question fragment ANYWHERE ("do you", "is there" …)
# Detector 4 — Tag question ANYWHERE ("right?", "isn't it", "don't you think")
# Detector 5 — Indirect/reported question patterns ("I wonder if", "tell me why")
# ─────────────────────────────────────────────────────────────────────────────

# D2 — question words that can open a sentence
_D2_OPENER = re.compile(
    r"(?:^|\.\s+|\?\s+|!\s+)"          # start or after sentence boundary
    r"(what|why|how|when|where|who|which|whose|whom"
    r"|can|could|would|should|shall|will|may|might|must"
    r"|is|are|was|were|do|does|did|have|has|had|am)\b",
    re.IGNORECASE,
)

# D3 — embedded question phrases that signal a question ANYWHERE in text
_D3_EMBEDDED = re.compile(
    r"\b("
    # auxiliary + subject combos
    r"do you|does he|does she|does it|do we|do they"
    r"|did you|did he|did she|did it|did we|did they"
    r"|are you|is he|is she|is it|are we|are they"
    r"|was he|was she|was it|were you|were we|were they"
    r"|have you|has he|has she|have we|have they"
    r"|had you|had he|had she|had we|had they"
    r"|can you|can he|can she|can we|can they"
    r"|could you|could he|could she|could we|could they"
    r"|would you|would he|would she|would we|would they"
    r"|should you|should he|should she|should we|should they"
    r"|will you|will he|will she|will we|will they"
    r"|shall we|shall I"
    r"|is there|are there|was there|were there"
    r"|is this|is that|are these|are those"
    r"|what is|what are|what was|what were|what do|what did|what have"
    r"|why is|why are|why do|why did|why would|why can|why should"
    r"|how is|how are|how do|how did|how can|how would|how should|how much|how many|how long|how often|how far"
    r"|when is|when are|when do|when did|when will|when would|when can"
    r"|where is|where are|where do|where did|where will|where would|where can"
    r"|who is|who are|who was|who were|who will|who would|who can|who did"
    r"|which is|which are|which was|which do|which did|which would"
    r")\b",
    re.IGNORECASE,
)

# D4 — tag questions that appear ANYWHERE (end or middle)
_D4_TAG = re.compile(
    r"\b("
    r"right\??|correct\??|yeah\??|ok\??"
    r"|isn'?t it|aren'?t you|wasn'?t it|weren'?t they"
    r"|don'?t you|doesn'?t it|didn'?t you|didn'?t they"
    r"|haven'?t you|hasn'?t it|hadn'?t you"
    r"|can'?t you|couldn'?t you|won'?t you|wouldn'?t you"
    r"|shouldn'?t it|shouldn'?t you|mustn'?t it"
    r"|do you think|don'?t you think|would you say"
    r"|you know\??"
    r")\b",
    re.IGNORECASE,
)

# D5 — indirect/reported question patterns (question embedded after a verb)
_D5_INDIRECT = re.compile(
    r"\b("
    r"I wonder (if|whether|what|why|how|when|where|who)"
    r"|tell me (what|why|how|when|where|who|if|whether)"
    r"|I('?m| am) not sure (if|whether|what|why|how)"
    r"|do you know (if|whether|what|why|how|when|where|who)"
    r"|I want to know (if|whether|what|why|how)"
    r"|can you (tell|explain|show|describe|clarify)"
    r"|could you (tell|explain|show|describe|clarify)"
    r"|would you (mind|like|be able to)"
    r"|I('?d| would) like to know"
    r"|any idea (if|whether|what|why|how|when|where|who)"
    r")\b",
    re.IGNORECASE,
)

# Greetings — never a question even if they start with "how"
_GREETINGS = re.compile(
    r"^(hello|hi|hey|good morning|good afternoon|good evening|greetings"
    r"|how are you|how('?s| is) it going|how('?s| is) everything)\b",
    re.IGNORECASE,
)


def _is_question(text: str) -> bool:
    """
    Returns True if ANY part of the text contains a question.
    Scans the FULL text — not just the start — so mid-sentence
    questions like 'I was thinking, but what do you mean?' are caught.
    """
    s = text.strip()
    if not s:
        return False

    # Greeting override (e.g. "How are you" is NOT a question here)
    if _GREETINGS.match(s):
        return False

    # D1 — explicit question mark anywhere
    if "?" in s:
        return True

    # D2 — question-opening word (handles sentence boundaries inside the chunk)
    if _D2_OPENER.search(s):
        return True

    # D3 — embedded auxiliary+subject combo anywhere
    if _D3_EMBEDDED.search(s):
        return True

    # D4 — tag question anywhere
    if _D4_TAG.search(s):
        return True

    # D5 — indirect question pattern anywhere
    if _D5_INDIRECT.search(s):
        return True

    return False


# ── SENTENCE SPLITTER ────────────────────────────────────────────────────────
# Google Speech rarely returns punctuation, so we use a smart splitter that
# detects clause boundaries from spoken discourse markers.

# Discourse transitions that almost always separate two independent clauses
_SPLIT_MARKERS = re.compile(
    r"\s+(?="
    r"(?:but (?:what|why|how|when|where|who|can|could|do|does|did|is|are|was|were|will|would|have|has))"
    r"|(?:so (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:and (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:now (?:what|why|how|when|where|who|can|could|do|does|did|is|are|will|would))"
    r"|(?:like (?:what|why|how|when|where|who))"
    r"|(?:i mean (?:what|why|how|when|where|who))"
    r")",
    re.IGNORECASE,
)

# Spoken question starters that clearly begin a new thought
_QUESTION_STARTERS = re.compile(
    r"\s+(?="
    r"(?:what|why|how|when|where|who|which)"
    r"\s+(?:is|are|was|were|do|does|did|can|could|will|would|should|have|has|had)\b"
    r")",
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> list:
    text = " ".join(text.split())

    # If Google gave us punctuation, honour it first
    if any(p in text for p in [".", "?", "!"]):
        parts = re.split(r"(?<=[.?!])\s+", text)
        # Then also split within each part on embedded question patterns
        final = []
        for p in parts:
            sub = _split_by_embedded_questions(p)
            final.extend(sub)
        return [s.strip() for s in final if s.strip()]

    # No punctuation path — split on discourse markers + embedded questions
    parts = re.split(
        r"\s+(?:and then|but then|however|also|because|after that"
        r"|okay so|alright so|now|anyway)\s+",
        text, flags=re.IGNORECASE,
    )

    final = []
    for p in parts:
        sub = _split_by_embedded_questions(p)
        final.extend(sub)

    return [s.strip() for s in final if s.strip()]


def _split_by_embedded_questions(text: str) -> list:
    """
    Further splits a chunk by detecting where a question begins mid-sentence.
    e.g. "I was talking about it but what do you think about that"
      → ["I was talking about it", "but what do you think about that"]
    """
    # Try discourse-marker-before-question-word splits
    parts = _SPLIT_MARKERS.split(text)
    if len(parts) == 1:
        # Try raw question-word-opener splits
        parts = _QUESTION_STARTERS.split(text)

    return [p.strip() for p in parts if p is not None and p.strip()]


def _format_sentence(sentence: str, question: bool) -> str:
    s = sentence.strip().rstrip(".?!")
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s + ("?" if question else ".")


def clean_text(text: str) -> list:
    """
    Split text into clauses and label each as question or statement.
    Returns: [{"sentence": "...", "is_question": bool}, ...]
    """
    sentences = _split_into_sentences(text)
    results = []
    for sent in sentences:
        q = _is_question(sent)
        results.append({"sentence": _format_sentence(sent, q), "is_question": q})
    return results


# ── 2. RECORD AUDIO (PUSH-TO-TALK) ────────────────────────────────────────────

def record_audio(trigger_event=None, mouth_active_event=None):
    import sounddevice as sd
    import numpy as np
    import queue
    import sys

    sample_rate = 16000  # 16kHz is standard for speech

    if trigger_event is not None:
        print("🎤 Waiting for voice detection from camera...")
        trigger_event.wait()
        trigger_event.clear()  # Reset for next time
    else:
        input("🎤 Press ENTER to START recording...")
        
    print("🔴 Recording... Speak now! Take your time. (will stop after 3s of silence or ~60s max)")

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    silence_threshold = 0.005  # More sensitive to quiet speech
    max_silence_frames = int(sample_rate * 3)  # Allow very long pauses so you can finish your thoughts
    max_total_frames = int(sample_rate * 60)  # Allow up to 60 seconds of continuous speech

    recorded_frames = []
    silence_frames = 0
    total_frames = 0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
        while True:
            data = q.get()
            recorded_frames.append(data)
            total_frames += len(data)

            # Check for silence in the current chunk
            if np.max(np.abs(data)) < silence_threshold:
                silence_frames += len(data)
            else:
                silence_frames = 0

            # SMART STOP LOGIC
            # If camera says mouth is closed, stop after 0.8s of silence.
            # If camera says mouth is open, wait up to 3s of silence.
            if mouth_active_event is not None and not mouth_active_event.is_set():
                dynamic_max_silence = int(sample_rate * 0.8)
            else:
                dynamic_max_silence = max_silence_frames

            if silence_frames > dynamic_max_silence:
                print("⏹ Silence detected. Recording finished.")
                break

            if total_frames >= max_total_frames:
                print("⏹ Max duration reached. Recording finished.")
                break

    if recorded_frames:
        audio_data = np.concatenate(recorded_frames, axis=0).flatten()
        # Strip trailing silence to clean the file up, keep 0.3s of it
        pad_frames = int(sample_rate * 0.3)
        trim_index = max(0, len(audio_data) - silence_frames + pad_frames)
        audio_data = audio_data[:trim_index]
    else:
        audio_data = np.zeros((0,), dtype='float32')

    return audio_data, sample_rate


# ── 3. CLEAN AUDIO (NOISE REDUCTION) ─────────────────────────────────────────

def clean_audio(audio_data, sr_rate):
    if audio_data.size == 0:
        print("🧹 No audio to clean.")
        return audio_data, sr_rate

    print("🧹 Reducing noise (fast pass)...")

    audio_data = np.asarray(audio_data).flatten().astype('float32')

    # Whisper is highly robust to noise. A single lightweight pass is enough
    # for stationary noise (e.g., AC hum or fan) and is 2x faster.
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr_rate, stationary=True, prop_decrease=0.6)

    # Normalise
    max_val = np.max(np.abs(reduced_noise))
    if max_val > 0:
        reduced_noise = reduced_noise / max_val * 0.95

    return reduced_noise, sr_rate


# ── 4. SPEECH → TEXT (WHISPER API) ────────────────────────────────────────────

def speech_to_text(audio, sr_rate):
    try:
        print("🧠 Transcribing with Whisper...")
        audio = np.asarray(audio).flatten().astype('float32')

        if audio.size == 0:
            return "❌ No audio to transcribe"

        model = get_whisper_model()
        # fp16=False suppresses warnings on CPU and prevents fallback overhead
        result = model.transcribe(audio, language="en", fp16=False)
        text = result["text"]

        if not text.strip():
            return "❌ Could not understand audio"

        return text

    except Exception as e:
        import traceback
        return f"❌ Transcription error: {e}\n{traceback.format_exc()}"


# ── 5. INTONATION HEURISTIC ───────────────────────────────────────────────────

def _ends_with_rising_pitch(audio, sr_rate):
    """Checks if energy rises at the end — rough signal for a question."""
    try:
        y = np.asarray(audio).flatten()
        if y.size == 0:
            return False
        half = int(0.5 * sr_rate)
        if len(y) < 2 * half:
            return False
        rms_end = np.sqrt(np.mean(y[-half:] ** 2))
        rms_pre = np.sqrt(np.mean(y[-2 * half:-half] ** 2))
        return rms_end > rms_pre * 1.15
    except Exception:
        return False


# ── 6. SAVE TO JSON ───────────────────────────────────────────────────────────

def save_to_json(sentences, raw_text, filename=None):
    if filename is None:
        filename = os.path.abspath("nlp_log.json")
    else:
        filename = os.path.abspath(filename)

    record = {
        "raw_text": raw_text,
        "sentences": sentences,
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(record)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"💾 Saved to {filename}")


# ── 7. FULL PIPELINE ──────────────────────────────────────────────────────────

def push_to_talk_pipeline(trigger_event=None, mouth_active_event=None):
    start_time = time.perf_counter()

    # Set working directory to script location for consistency
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Step 1: Record
    raw_audio, raw_sr = record_audio(trigger_event, mouth_active_event)

    # Step 2: Clean audio
    clean_audio_arr, clean_sr = clean_audio(raw_audio, raw_sr)

    # Step 3: Speech → text
    raw_text = speech_to_text(clean_audio_arr, clean_sr)

    # Step 4: Show raw output
    print("\n" + "=" * 40)
    print("🧠 RAW TEXT:")
    print(raw_text)

    rising = _ends_with_rising_pitch(clean_audio_arr, clean_sr)
    if rising:
        print("📈 (Intonation hint: rising energy detected at end)")
    print("=" * 40)

    if not raw_text or "❌" in raw_text:
        print("⚠️ Ignored invalid or unclear audio (not saved).")
        elapsed = time.perf_counter() - start_time
        print(f"⏱️ Run time: {elapsed:.2f} seconds")
        return []

    # Step 5: Split into sentences + detect questions
    sentences = clean_text(raw_text)

    print("\n✨ DETECTED SENTENCES:")
    for i, item in enumerate(sentences, 1):
        label = "❓ QUESTION" if item["is_question"] else "💬 Statement"
        print(f"  [{i}] {label}: {item['sentence']}")
    print("=" * 40 + "\n")

    # Step 6: Save
    if sentences:
        save_to_json(sentences, raw_text)
    else:
        print("⚠️ Nothing to save.")

    elapsed = time.perf_counter() - start_time
    print(f"⏱️ Total run time: {elapsed:.2f} seconds")
    return sentences


# ── 8. RUN SYSTEM ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- 🎓 Smart Classroom Speech Module ---\n")

    while True:
        push_to_talk_pipeline()

        choice = input("Press Q to quit or ENTER to record again: ")
        if choice.lower() == "q":
            print("👋 Exiting system...")
            break