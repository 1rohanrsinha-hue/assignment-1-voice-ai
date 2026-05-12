import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"

import tempfile
import time
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import ollama
from kokoro import KPipeline
import soundfile as sf

# Faster Whisper
whisper_model = WhisperModel("tiny", device="cuda")

# Kokoro
pipeline = KPipeline(lang_code='a')

fs = 16000

print("Full Voice AI started.")
print("Speak into the mic.")
print("Press CTRL+C to stop.\n")

while True:
    print("Listening...")

    recording = sd.rec(int(3 * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        write(temp_audio.name, fs, recording)

        segments, info = whisper_model.transcribe(temp_audio.name)

        user_text = " ".join([segment.text for segment in segments]).strip()

        print(f"\nYou said: {user_text}")

        if len(user_text.strip()) > 3:

            response = ollama.chat(
                model="llama3:8b",
                messages=[
                    {
                        "role": "system",
                        "content": "Respond briefly in one sentence."
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ]
            )

            ai_reply = response["message"]["content"]

            print(f"\nAI: {ai_reply}")

            generator = pipeline(
                ai_reply,
                voice="af_heart"
            )

            for i, (gs, ps, audio) in enumerate(generator):
                output_file = "reply.wav"
                sf.write(output_file, audio, 24000)

            os.system("start reply.wav")

            time.sleep(5)

            print("\n-------------------\n")