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
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
SILENCE_CHUNKS = 10        # ~2s of silence = end of utterance
MIN_SPEECH_CHUNKS = 3      # ignore clips shorter than ~600ms


class MicListener:
    """Stream mic audio, detect utterances, transcribe, yield text."""

    def __init__(self, model_size: str = "tiny") -> None:
        self._model_size = model_size
        self._model = None
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()

    def _load_model(self) -> None:
        if self._model is None:
            log.info("Loading Whisper model '%s' (first run downloads ~75MB)…", self._model_size)
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            log.info("Whisper ready.")

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        return text

    def _mic_callback(self, indata, frames, time, status) -> None:
        if status:
            log.debug("Mic status: %s", status)
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
                elif speaking:
                    silence_count += 1
                    buffer.append(chunk)
                    if silence_count >= SILENCE_CHUNKS:
                        if len(buffer) >= MIN_SPEECH_CHUNKS:
                            audio = np.concatenate(buffer)
                            text = self._transcribe(audio)
                            if text:
                                yield text
                        buffer.clear()
                        silence_count = 0
                        speaking = False

    def stop(self) -> None:
        self._stop.set()

