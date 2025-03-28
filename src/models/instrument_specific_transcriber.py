import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Union
from typing import Tuple  # Import separately to avoid errors
import random

class InstrumentSpecificTranscriber(nn.Module):
    """Instrument-specific transcription model"""
    
    def __init__(
        self,
        instrument: str,
        input_dim: int = 64,  # Matches our optimized mel bins
        hidden_dim: int = 128,
        num_pitches: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize an instrument-specific transcription model
        
        Args:
            instrument: The instrument name
            input_dim: Input feature dimension (mel bands)
            hidden_dim: Hidden dimension size
            num_pitches: Number of pitch classes
            dropout: Dropout rate
        """
        super().__init__()
        self.instrument = instrument
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_pitches = num_pitches
        
        # Input projection layer to handle variable sized inputs
        self.input_projection = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )
        
        # Adaptive pooling to ensure consistent tensor dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((input_dim, 128))
        
        # Spectral feature processing blocks
        self.spectral_blocks = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(
            64 * (input_dim // 4), hidden_dim, kernel_size=3, padding=1
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pitches)
        )
        
        self.instrument_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Thresholds (can be learned or manually set)
        self.onset_threshold = 0.5
        self.pitch_threshold = 0.3
        self.instrument_threshold = 0.5
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, n_mels, time]
                or [batch_size, channels, n_mels, time]
                
        Returns:
            Tuple of (onset_logits, pitch_logits, instrument_logits)
        """
        batch_size = x.shape[0]
        
        # Handle different input formats
        if x.dim() == 3:  # [batch_size, n_mels, time]
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, n_mels, time]
        
        # Ensure 4D input [batch_size, channels, height, width]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got shape {x.shape}")
            
        # Apply adaptive pooling to ensure consistent dimensions
        x = self.adaptive_pool(x)
        
        # Apply input projection
        x = self.input_projection(x)
        
        # Process through spectral blocks
        x = self.spectral_blocks(x)
        
        # Reshape for temporal processing
        # [batch_size, channels*height, width]
        x = x.reshape(batch_size, -1, x.shape[-1])
        
        # Apply temporal convolution
        x = F.relu(self.temporal_conv(x))
        
        # Global average pooling over time
        time_dim = x.shape[-1]
        onset_logits = torch.zeros(batch_size, time_dim).to(x.device)
        pitch_logits = torch.zeros(batch_size, time_dim, self.num_pitches).to(x.device)
        instrument_logits = torch.zeros(batch_size, time_dim).to(x.device)
        
        # Process each time step
        for t in range(time_dim):
            h_t = x[:, :, t]
            
            # Apply output heads
            onset_logits[:, t] = self.onset_head(h_t).squeeze(-1)
            pitch_logits[:, t] = self.pitch_head(h_t)
            instrument_logits[:, t] = self.instrument_head(h_t).squeeze(-1)
        
        return onset_logits, pitch_logits, instrument_logits
    
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make predictions from input features
        
        Args:
            features: Dictionary of input features
                
        Returns:
            Dictionary of predictions
        """
        # Get mel spectrogram from features
        mel_spec = features["mel_spectrogram"]
        
        # Handle feature dimensions
        if mel_spec.dim() == 2:  # [n_mels, time]
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        onset_logits, pitch_logits, instrument_logits = self.forward(mel_spec)
        
        # Apply sigmoid to get probabilities
        onset_probs = torch.sigmoid(onset_logits)
        pitch_probs = torch.softmax(pitch_logits, dim=-1)
        instrument_probs = torch.sigmoid(instrument_logits)
        
        return {
            "onset_probs": onset_probs,
            "pitch_probs": pitch_probs,
            "instrument_probs": instrument_probs
        }
    
    def find_onset_peaks(self, onset_probs: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Find peaks in onset probability curve
        
        Args:
            onset_probs: Onset probabilities [time]
            threshold: Detection threshold (default: self.onset_threshold)
                
        Returns:
            Tensor of frame indices where onsets occur
        """
        if threshold is None:
            threshold = self.onset_threshold
        
        # Detect peaks
        peaks = []
        for i in range(1, len(onset_probs) - 1):
            if (onset_probs[i] > threshold and
                onset_probs[i] > onset_probs[i-1] and
                onset_probs[i] > onset_probs[i+1]):
                peaks.append(i)
        
        return torch.tensor(peaks)
    
    def refine_pitch(self, pitch_probs: torch.Tensor, onset_times: torch.Tensor) -> torch.Tensor:
        """
        Refine pitch predictions at onset times
        
        Args:
            pitch_probs: Pitch class probabilities [time, num_pitches]
            onset_times: Frame indices of onset times
                
        Returns:
            Tensor of pitch indices at onset times
        """
        pitches = []
        
        for t in onset_times:
            t = t.item()
            # Find the most likely pitch at this time
            pitch = torch.argmax(pitch_probs[t]).item()
            pitches.append(pitch)
        
        return torch.tensor(pitches)
    
    def transcribe(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Transcribe audio features
        
        Args:
            features: Dictionary of input features
                
        Returns:
            Dictionary of note events
        """
        with torch.no_grad():
            # Get model predictions
            predictions = self.predict(features)
            
            # Extract probabilities
            onset_probs = predictions["onset_probs"].squeeze(0).cpu()
            pitch_probs = predictions["pitch_probs"].squeeze(0).cpu()
            instrument_probs = predictions["instrument_probs"].squeeze(0).cpu()
            
            # Find onsets
            onset_peaks = self.find_onset_peaks(onset_probs)
            
            # Early return for empty sequence
            if len(onset_peaks) == 0:
                return {
                    "onset_times": [],
                    "pitches": [],
                    "instruments": []
                }
            
            # Get pitches at onset times
            pitches = self.refine_pitch(pitch_probs, onset_peaks)
            
            # Create note events
            return {
                "onset_times": onset_peaks,
                "pitches": pitches,
                "instruments": [self.instrument] * len(onset_peaks)
            } 