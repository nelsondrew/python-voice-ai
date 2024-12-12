from TTS.api import TTS
from io import BytesIO
import asyncio

# Global variables for caching
tts_model = None
model_loaded = False
device = 'cpu'

# Function to load the TTS model once and reuse it
def get_tts_model():
    global tts_model, model_loaded, device
    if not model_loaded:
        print("Loading the TTS model...")
        tts_model = TTS(model_name="tts_models/en/ljspeech/fast_pitch").to(device)
        model_loaded = True
    return tts_model

# Function to synthesize speech asynchronously
async def synthesize_text_async(text: str) -> BytesIO:
    tts_model = get_tts_model()  # Use the cached model
    # Create a temporary BytesIO buffer to save the audio to
    audio_buffer = BytesIO()
    # Save audio to buffer asynchronously
    await asyncio.to_thread(tts_model.tts_to_file, text, file_path=audio_buffer)
    audio_buffer.seek(0)  # Rewind the buffer to the start
    return audio_buffer

# Function to generate speech from text and return audio response
async def generate_speech_from_text(text: str) -> bytes:
    audio_buffer = await synthesize_text_async(text)
    return audio_buffer.read()  # Return the raw audio bytes (expected in main.py)
