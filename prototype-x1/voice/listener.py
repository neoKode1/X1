"""
Always-on microphone listener for ARIA.
Captures audio, detects silence boundaries, transcribes with faster-whisper.
Yields transcribed text chunks for the main loop to process.
"""
from __future__ import annotations
import logging
import queue
import threading
import numpy as np

log = logging.getLogger("x1.listener")

SAMPLE_RATE = 16000
CHUNK_MS = 200          # ms per audio chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
SILENCE_THRESHOLD = 0.025  # RMS below this = silence (raised from 0.01 to reject ambient noise)
SILENCE_CHUNKS = 8         # ~1.6s of silence = end of utterance (faster cutoff)
MIN_SPEECH_CHUNKS = 4      # ignore clips shorter than ~800ms
MAX_BUFFER_CHUNKS = 50     # cap at ~10s — prevents 30s+ noise accumulation

# Whisper-tiny hallucinates these phrases on ambient noise / silence
_HALLUCINATION_PHRASES = {
    "thanks for watching", "subscribe", "like and subscribe",
    "thank you for watching", "you got poop", "i'm not a fool",
    "please subscribe", "see you next time", "bye bye",
    "subs by", "subtitles by", "amara.org", "you",
    "i'm sorry", "birth certificate", "taxid", "tax id",
    "i'm going to be a physician", "i'm not going to be a",
}

import re as _re

def _is_hallucination(text: str) -> bool:
    """Reject known Whisper hallucination patterns."""
    t = text.lower().strip().rstrip(".")
    if len(t) < 3:
        return True
    if t in _HALLUCINATION_PHRASES:
        return True
    for phrase in _HALLUCINATION_PHRASES:
        if t == phrase or t.startswith(phrase):
            return True
    # Reject pure repetition: "word word word word"
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    # Reject repeated phrases: "I'm sorry. I'm sorry. I'm sorry."
    # Split on sentence boundaries and check for repeated sentences
    sentences = [s.strip().rstrip(".!?").strip() for s in _re.split(r'[.!?]+', t) if s.strip()]
    if len(sentences) >= 2:
        unique = set(sentences)
        if len(unique) == 1:
            return True
        # >60% of sentences are the same phrase — likely hallucination
        from collections import Counter
        counts = Counter(sentences)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(sentences) > 0.6:
            return True
    # Reject if >70% of words are unique but the text makes no coherent sense
    # (heuristic: very long transcriptions from silence tend to be nonsensical)
    if len(words) > 15 and len(sentences) >= 3:
        return True  # ambient noise rarely produces 15+ real words
    return False


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}" if m else f"00:{s:05.2f}"


class MicListener:
    """Stream mic audio, detect utterances, transcribe, yield text."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = None
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self.muted = threading.Event()  # set this while ARIA is speaking to avoid echo

    def _load_model(self) -> None:
        if self._model is None:
            log.info("Loading Whisper model '%s' (first run downloads ~75MB)…", self._model_size)
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            log.info("Whisper ready.")

    # Domain vocabulary for Whisper initial_prompt bias
    _VOCAB_PROMPT = (
        "ARIA, Vibcoder, Neokode, Chad, BCS, cyberpunk, mods, "
        "screengrab, Ollama, Llama, prototype, locomotion, "
        "Raspberry Pi, servo, actuator, neural, chassis"
    )

    # Post-processing corrections for common Whisper mishearings
    _WORD_CORRECTIONS = {
        "mim coder": "Vibcoder", "mim coater": "Vibcoder",
        "vibe coder": "Vibcoder", "vib coder": "Vibcoder",
        "vibecoder": "Vibcoder", "vibecoda": "Vibcoder",
        "iapx": "Vibcoder", "vip coder": "Vibcoder",
        "vim coder": "Vibcoder", "by coder": "Vibcoder",
        "aria": "ARIA", "arya": "ARIA", "area": "ARIA",
        "neo code": "Neokode", "neo kode": "Neokode",
        "neocode": "Neokode",
    }

    def _post_correct(self, text: str) -> str:
        """Fix common Whisper mishearings via case-insensitive replacement."""
        result = text
        lower = result.lower()
        for wrong, right in self._WORD_CORRECTIONS.items():
            idx = lower.find(wrong)
            while idx != -1:
                result = result[:idx] + right + result[idx + len(wrong):]
                lower = result.lower()
                idx = lower.find(wrong, idx + len(right))
        return result

    def _transcribe(self, audio: np.ndarray) -> str:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            segments, _ = self._model.transcribe(
                audio, language="en", beam_size=3,
                initial_prompt=self._VOCAB_PROMPT,
            )
            text = " ".join(s.text for s in segments).strip()
        return self._post_correct(text)

    def _mic_callback(self, indata, frames, time, status) -> None:
        if status:
            log.debug("Mic status: %s", status)
        if not self.muted.is_set():
            self._q.put(indata[:, 0].copy())

    def listen(self):
        """Generator — yields transcribed utterance strings."""
        import sounddevice as sd
        self._load_model()
        log.info("Mic open — listening.")

        buffer: list[np.ndarray] = []
        silence_count = 0
        speaking = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=CHUNK_SAMPLES,
                            callback=self._mic_callback):
            while not self._stop.is_set():
                try:
                    chunk = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                is_speech = rms > SILENCE_THRESHOLD

                if is_speech:
                    buffer.append(chunk)
                    silence_count = 0
                    speaking = True
                    # Cap buffer to prevent 30s+ noise accumulation
                    if len(buffer) > MAX_BUFFER_CHUNKS:
                        audio = np.concatenate(buffer)
                        rms_total = float(np.sqrt(np.mean(audio ** 2)))
                        if rms_total > SILENCE_THRESHOLD:
                            log.info("Processing audio with duration %s (capped)",
                                     _fmt_dur(len(audio) / SAMPLE_RATE))
                            text = self._transcribe(audio)
                            if text and not _is_hallucination(text):
                                log.info("Mic: %s", text)
                                yield text
                        buffer.clear()
                        silence_count = 0
                        speaking = False
                elif speaking:
                    silence_count += 1
                    buffer.append(chunk)
                    if silence_count >= SILENCE_CHUNKS:
                        if len(buffer) >= MIN_SPEECH_CHUNKS:
                            audio = np.concatenate(buffer)
                            rms_total = float(np.sqrt(np.mean(audio ** 2)))
                            if rms_total > SILENCE_THRESHOLD:
                                log.info("Processing audio with duration %s",
                                         _fmt_dur(len(audio) / SAMPLE_RATE))
                                text = self._transcribe(audio)
                                if text and not _is_hallucination(text):
                                    log.info("Mic: %s", text)
                                    yield text
                        buffer.clear()
                        silence_count = 0
                        speaking = False

    def stop(self) -> None:
        self._stop.set()

