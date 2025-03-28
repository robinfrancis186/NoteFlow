from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import torch
import tempfile
import os
from pathlib import Path
import shutil
import time
from typing import Optional, List
import numpy as np
from pydantic import BaseModel

from src.preprocessing.audio_processor import AudioProcessor
from src.models.transcriber import MusicTranscriber
from src.postprocessing.midi_generator import MIDIGenerator

class TranscriptionRequest(BaseModel):
    tempo: Optional[float] = None
    onset_threshold: float = 0.5
    pitch_threshold: float = 0.5
    instrument_threshold: float = 0.3

class TranscriptionStats(BaseModel):
    processing_time: float
    audio_length: float
    num_instruments_detected: int
    detected_instruments: List[str]
    estimated_tempo: float
    time_signature: str

app = FastAPI(
    title="AI Music Transcription API",
    description="Competition-ready music transcription system",
    version="1.0.0"
)

# Initialize components
audio_processor = AudioProcessor()
transcriber = MusicTranscriber()
midi_generator = MIDIGenerator()

# Load model weights if available
MODEL_PATH = Path("models/transcriber.pt")
if MODEL_PATH.exists():
    transcriber.load_state_dict(torch.load(MODEL_PATH))
transcriber.eval()

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transcriber = transcriber.to(device)

@app.post("/transcribe/", response_model=TranscriptionStats)
async def transcribe_audio(
    file: UploadFile = File(...),
    params: Optional[TranscriptionRequest] = None
):
    """Transcribe uploaded audio file to MIDI
    
    Competition requirements:
    - Process within 6 seconds per file
    - Handle up to 3 instruments
    - Support specific tempo range (60-90 BPM)
    - Handle recordings up to 20 seconds
    """
    if params is None:
        params = TranscriptionRequest()
        
    try:
        start_time = time.time()
        
        # Validate file size and duration
        file_size = 0
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in file.file:
                file_size += len(chunk)
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    raise HTTPException(
                        status_code=400,
                        detail="File too large. Maximum size is 10MB."
                    )
                temp_file.write(chunk)
                
            temp_path = temp_file.name
            
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Preprocess audio
            features = audio_processor.preprocess_for_model(temp_path)
            
            # Calculate audio length
            audio_length = len(features["harmonic"]) / audio_processor.sample_rate
            
            if audio_length > 20.0:
                raise HTTPException(
                    status_code=400,
                    detail="Audio file too long. Maximum duration is 20 seconds."
                )
            
            # Move features to device
            features = {k: v.to(device) for k, v in features.items()}
            
            # Get model predictions
            with torch.no_grad():
                predictions = transcriber.transcribe(
                    features,
                    onset_threshold=params.onset_threshold,
                    pitch_threshold=params.pitch_threshold,
                    instrument_threshold=params.instrument_threshold
                )
                
            # Generate MIDI
            midi_data = midi_generator.predictions_to_midi(
                predictions,
                tempo=params.tempo
            )
            
            # Save MIDI file
            temp_midi = Path(temp_dir) / "output.mid"
            midi_data.write(str(temp_midi))
            
            # Calculate statistics
            processing_time = time.time() - start_time
            
            if processing_time > 6.0:
                print(f"Warning: Processing time ({processing_time:.2f}s) "
                      f"exceeded competition limit of 6.0s")
            
            # Get detected time signature
            time_sig = midi_data.time_signatures[0]
            time_sig_str = f"{time_sig.numerator}/{time_sig.denominator}"
            
            # Prepare response
            stats = TranscriptionStats(
                processing_time=processing_time,
                audio_length=audio_length,
                num_instruments_detected=len(predictions["active_instruments"]),
                detected_instruments=predictions["active_instruments"],
                estimated_tempo=midi_data.estimate_tempo(),
                time_signature=time_sig_str
            )
            
            # Return both MIDI file and stats
            return FileResponse(
                temp_midi,
                media_type="audio/midi",
                filename="transcription.mid",
                headers={"X-Transcription-Stats": stats.json()}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            os.unlink(temp_path)
    
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": MODEL_PATH.exists()
    }

@app.get("/competition-ready")
def competition_check():
    """Check if the system meets competition requirements"""
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": MODEL_PATH.exists(),
        "supported_instruments": list(midi_generator.instrument_programs.keys()),
        "tempo_range": f"{midi_generator.min_tempo}-{midi_generator.max_tempo} BPM",
        "pitch_range": f"C{midi_generator.min_pitch//12}-C{midi_generator.max_pitch//12}",
        "time_signatures": [f"{n}/{d}" for n, d in midi_generator.supported_time_sigs]
    }
    
    all_passed = all([
        checks["cuda_available"],
        checks["model_loaded"],
        len(checks["supported_instruments"]) == 8
    ])
    
    return {
        "ready": all_passed,
        "checks": checks
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 