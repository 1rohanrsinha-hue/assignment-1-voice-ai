import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"

import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

model = whisper.load_model("base")

fs = 16000

print("Live Whisper started. Speak into the mic.")
print("Press CTRL+C to stop.\n")

while True:
    print("Recording...")

    recording = sd.rec(int(5 * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        write(temp_audio.name, fs, recording)

        result = model.transcribe(temp_audio.name)

        print("\nYou said:")
        print(result["text"])
        print("\n---\n")