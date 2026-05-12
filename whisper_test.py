import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"

import whisper

model = whisper.load_model("base")

result = model.transcribe("test.wav")

print(result["text"])