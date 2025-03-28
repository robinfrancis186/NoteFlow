import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from src.preprocessing.audio_processor import AudioProcessor
from src.models.transcriber import MusicTranscriber, OnsetDetector, PitchEstimator
from src.postprocessing.midi_generator import MIDIGenerator, Note

@pytest.fixture
def audio_processor():
    return AudioProcessor()

@pytest.fixture
def transcriber():
    return MusicTranscriber()

@pytest.fixture
def midi_generator():
    return MIDIGenerator()

@pytest.fixture
def sample_audio():
    """Create a simple sine wave for testing"""
    sample_rate = 44100
    duration = 2.0  # seconds
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio, sample_rate)
        return Path(temp_file.name)

def test_audio_processor(audio_processor, sample_audio):
    """Test audio preprocessing"""
    # Test loading audio
    audio, sr = audio_processor.load_audio(str(sample_audio))
    assert sr == audio_processor.sample_rate
    assert len(audio.shape) == 1
    
    # Test feature extraction
    features = audio_processor.extract_features(audio)
    assert features.mel_spectrogram.shape[1] > 0
    assert len(features.harmonic) == len(audio)
    assert len(features.percussive) == len(audio)
    
def test_onset_detector():
    """Test onset detection model"""
    model = OnsetDetector()
    batch_size = 2
    time_steps = 100
    input_tensor = torch.randn(batch_size, 1, time_steps, 128)
    
    output = model(input_tensor)
    assert output.shape == (batch_size, time_steps, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_pitch_estimator():
    """Test pitch estimation model"""
    model = PitchEstimator()
    batch_size = 2
    time_steps = 100
    input_tensor = torch.randn(batch_size, time_steps, 128)
    
    output = model(input_tensor)
    assert output.shape == (batch_size, time_steps, 88)  # 88 piano keys
    assert torch.all((output >= 0) & (output <= 1))

def test_midi_generator(midi_generator):
    """Test MIDI generation"""
    # Create some test notes
    notes = [
        Note(pitch=60, start_time=0.0, end_time=1.0),  # Middle C
        Note(pitch=64, start_time=1.0, end_time=2.0),  # E
        Note(pitch=67, start_time=2.0, end_time=3.0)   # G
    ]
    
    # Test timing correction
    quantized_notes = midi_generator.apply_timing_correction(notes)
    assert len(quantized_notes) == len(notes)
    
    # Test MIDI creation
    midi_data = midi_generator.create_midi(notes)
    assert len(midi_data.instruments) == 1
    assert len(midi_data.instruments[0].notes) == len(notes)

def test_full_pipeline(audio_processor, transcriber, midi_generator, sample_audio):
    """Test complete transcription pipeline"""
    # Preprocess audio
    features = audio_processor.preprocess_for_model(str(sample_audio))
    
    # Get predictions
    with torch.no_grad():
        predictions = transcriber.transcribe(features)
    
    # Generate MIDI
    midi_data = midi_generator.predictions_to_midi(predictions)
    
    # Basic assertions
    assert "onset_times" in predictions
    assert "pitches" in predictions
    assert len(midi_data.instruments) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 