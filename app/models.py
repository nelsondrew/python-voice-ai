import multiprocessing
import tempfile
import concurrent
from typing import List
import openai
import torch
import os
from faster_whisper import WhisperModel
from pydub import AudioSegment

# Parallel processing optimization
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)

# Ultra-lightweight model initialization
MODEL_SIZES = ["base"]  # Single model for simplicity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Whisper model
model = WhisperModel(
    MODEL_SIZES[0],
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "float32",
    cpu_threads=NUM_CORES
)

def split_audio(audio_path, segment_duration=10):
    """
    Split an audio file into fixed-duration segments.
    
    Args:
        audio_path (str): Path to the audio file.
        segment_duration (int): Duration of each segment in seconds.
    
    Returns:
        list[AudioSegment]: List of audio segments.
    """
    audio = AudioSegment.from_file(audio_path)
    audio_length = len(audio) / 1000  # Convert to seconds
    
    segments = []
    for start_time in range(0, int(audio_length), segment_duration):
        end_time = min(start_time + segment_duration, audio_length)
        segment = audio[start_time * 1000:end_time * 1000]  # Convert to milliseconds
        segments.append(segment)
    return segments

def transcribe_segment(segment, language="en"):
    """
    Transcribe a specific audio segment.
    
    Args:
        segment (AudioSegment): Audio segment to transcribe.
        language (str, optional): Language of the audio.
    
    Returns:
        str: Transcribed text for the segment.
    """
    try:
        # Save segment to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_segment_file:
            segment.export(temp_segment_file.name, format="wav")
            
            # Transcribe the audio file
            segments, _ = model.transcribe(
                temp_segment_file.name,
                beam_size=1,
                language=language,
                condition_on_previous_text=False
            )
            return " ".join(segment.text.strip() for segment in segments)
    except Exception as e:
        print(f"Segment transcription error: {e}")
        return ""

def transcribe_audio(audio_data: bytes, language: str = "en") -> str:
    """
    Optimized audio transcription with proper segment handling.
    
    Args:
        audio_data (bytes): Raw audio file bytes.
        language (str, optional): Language of the audio. Defaults to "en".
    
    Returns:
        str: Transcribed text.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
        try:
            # Write audio data to temporary file
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            # Split the audio into non-overlapping segments
            audio_segments = split_audio(temp_audio.name, segment_duration=10)
            
            # Parallel transcription of segments
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
                futures = [
                    executor.submit(transcribe_segment, segment, language)
                    for segment in audio_segments
                ]
                # Collect results
                transcriptions = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Combine transcriptions
            final_transcription = " ".join(filter(bool, transcriptions))
            return final_transcription or ""
        
        except Exception as e:
            print(f"Parallel transcription error: {e}")
            return ""

def get_gpt_response(prompt: List[dict]) -> str:
    """Generate a response from OpenAI GPT-3.5."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=4096,
        n=1,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
