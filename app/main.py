from fastapi import FastAPI, File, UploadFile
import time
from io import BytesIO
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# Import functions from models.py and tts.py
from app.conversation_context import ConversationContext
from app.models import transcribe_audio, get_gpt_response
from app.tts import generate_speech_from_text  # Import the new function

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FastAPI server is running...")


conversation_context = ConversationContext()


@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    # Measure the start time for total latency calculation
    start_time = time.time()
    log_times = {}  # Dictionary to store timestamps for each step

    # Step 1: Read the uploaded audio file
    step_start = time.time()
    audio_data = await file.read()
    log_times["Read Audio File"] = time.time() - step_start
    print(f"Step 1: Read Audio File - {log_times['Read Audio File']:.2f} seconds")

    # Step 2: Transcribe audio to text using Whisper
    step_start = time.time()
    transcription = transcribe_audio(audio_data)
    log_times["Transcription"] = time.time() - step_start
    print(f"Step 2: Transcription - {log_times['Transcription']:.2f} seconds")
    print(f"Transcription: {transcription}")
    
    conversation_context.add_message("user", transcription)

    # Step 3: Generate GPT-3.5 response
    current_context = conversation_context.get_messages()
    print("Current context" , current_context)
    step_start = time.time()
    gpt_response = get_gpt_response(current_context)
    log_times["Generate GPT Response"] = time.time() - step_start
    print(f"Step 3: Generate GPT Response - {log_times['Generate GPT Response']:.2f} seconds")
    print(f"GPT-3.5 Response: {gpt_response}")

    # Step 4: Convert GPT-3.5 response to speech using TTS
    step_start = time.time()
    tts_audio = await generate_speech_from_text(gpt_response)
    log_times["Text-to-Speech Conversion"] = time.time() - step_start
    print(f"Step 4: Text-to-Speech Conversion - {log_times['Text-to-Speech Conversion']:.2f} seconds")

    # Measure total processing time
    total_processing_time = time.time() - start_time
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")

    # Return the audio response as a streaming response
    return StreamingResponse(BytesIO(tts_audio), media_type="audio/mpeg", headers={"Content-Disposition": "attachment; filename=output.mp3"})
