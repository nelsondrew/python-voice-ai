import tempfile
import wave
from io import BytesIO

def save_audio_file(audio_data: bytes) -> str:
    """Save the incoming audio data to a temporary file."""
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data)
    return temp_audio_path

def read_audio_file(file_path: str) -> BytesIO:
    """Read the audio file and return a BytesIO object."""
    with open(file_path, "rb") as f:
        audio_data = BytesIO(f.read())
    return audio_data

def get_audio_duration(file_path: str) -> float:
    """Get the duration of the audio file in seconds."""
    with wave.open(file_path, 'rb') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
    return duration
