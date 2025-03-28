import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import os
import librosa
from pathlib import Path

from .instrument_specific_transcriber import InstrumentSpecificTranscriber

class EnsembleTranscriber:
    """Ensemble of instrument-specific transcribers"""
    
    def __init__(
        self,
        instruments: List[str],
        sample_rate: int = 44100,
        hidden_dim: int = 128,
        device: str = "cpu",
        chunk_size: int = 4096
    ):
        """
        Initialize an ensemble of instrument-specific transcribers
        
        Args:
            instruments: List of instruments to include in the ensemble
            sample_rate: Audio sample rate
            hidden_dim: Hidden dimension for models
            device: Device to run models on
            chunk_size: Size of audio chunks for processing
        """
        self.instruments = instruments
        self.sample_rate = sample_rate
        self.device = device
        self.chunk_size = chunk_size
        
        # Set thresholds for onset, pitch, and instrument detection
        self.onset_threshold = 0.3
        self.pitch_threshold = 0.2
        self.instrument_threshold = 0.3
        
        # Initialize models for each instrument
        self.models = {}
        for instrument in instruments:
            self.models[instrument] = InstrumentSpecificTranscriber(
                instrument=instrument,
                input_dim=64,  # Matches optimized AudioProcessor
                hidden_dim=hidden_dim,
                num_pitches=128,
                dropout=0.3
            ).to(device)
            
        # Set models to evaluation mode
        for model in self.models.values():
            model.eval()
    
    def transcribe(self, features: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Transcribe audio features using all instrument models
        
        Args:
            features: Dictionary of input features
                
        Returns:
            Dictionary of predictions for each instrument
        """
        results = {}
        
        # Ensure features are on the correct device
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Run inference for each instrument model
        start_time = time.time()
        for instrument, model in self.models.items():
            # Skip if we've spent too much time already (more than 5 seconds)
            # This ensures we don't exceed competition time limits
            if time.time() - start_time > 5.0:
                print(f"Skipping {instrument} due to time constraints")
                continue
                
            try:
                with torch.no_grad():
                    results[instrument] = model.predict(features)
            except Exception as e:
                print(f"Error processing {instrument}: {str(e)}")
                # Return empty predictions on error
                results[instrument] = {
                    "onset_probs": torch.zeros(1, 100).to(self.device),
                    "pitch_probs": torch.zeros(1, 100, 128).to(self.device),
                    "instrument_probs": torch.zeros(1, 100).to(self.device)
                }
        
        return results
    
    def _adjust_thresholds(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Dynamically adjust thresholds based on audio characteristics
        
        Args:
            features: Dictionary of input features
        """
        # Get mel spectrogram
        mel_spec = features["mel_spectrogram"]
        
        # Compute energy of the signal
        if mel_spec.dim() > 2:
            energy = torch.mean(torch.abs(mel_spec.squeeze(0)))
        else:
            energy = torch.mean(torch.abs(mel_spec))
        
        # Adjust thresholds based on energy
        if energy < 0.1:  # Low energy/quiet recording
            # Lower thresholds to detect quiet sounds
            self.onset_threshold = 0.2
            self.pitch_threshold = 0.15
            self.instrument_threshold = 0.2
        elif energy > 0.5:  # High energy/loud recording
            # Raise thresholds to avoid false positives
            self.onset_threshold = 0.4
            self.pitch_threshold = 0.3
            self.instrument_threshold = 0.4
        else:  # Medium energy
            # Default thresholds
            self.onset_threshold = 0.3
            self.pitch_threshold = 0.2
            self.instrument_threshold = 0.3
            
        # Apply threshold updates to models
        for model in self.models.values():
            model.onset_threshold = self.onset_threshold
            model.pitch_threshold = self.pitch_threshold
            model.instrument_threshold = self.instrument_threshold

    def load_models(self, model_dir: str) -> None:
        """Load model weights from directory"""
        model_dir = Path(model_dir)
        for instrument in self.models:
            model_path = model_dir / f"{instrument}_transcriber.pt"
            if model_path.exists():
                self.models[instrument].load_state_dict(
                    torch.load(model_path, map_location=self.device)
                ) 