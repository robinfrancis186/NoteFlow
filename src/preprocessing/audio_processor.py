import numpy as np
import librosa
import torch
import torchaudio
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mel_spectrogram: np.ndarray
    harmonic: np.ndarray
    percussive: np.ndarray
    sample_rate: int
    onset_envelope: np.ndarray

class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        """Initialize audio processor with configurable parameters
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel bands
            hop_length: Number of samples between successive frames
            n_fft: Length of the FFT window
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and resample audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def extract_features(self, audio: np.ndarray) -> AudioFeatures:
        """Extract all relevant features from audio
        
        Args:
            audio: Audio time series
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Harmonic-percussive source separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return AudioFeatures(
            mel_spectrogram=mel_spec_db,
            harmonic=harmonic,
            percussive=percussive,
            sample_rate=self.sample_rate,
            onset_envelope=onset_env
        )
    
    def apply_augmentations(
        self, 
        audio: np.ndarray,
        noise_level: float = 0.005,
        pitch_shift: int = None
    ) -> np.ndarray:
        """Apply audio augmentations for training
        
        Args:
            audio: Input audio
            noise_level: Amount of noise to add (0-1)
            pitch_shift: Number of semitones to shift pitch
            
        Returns:
            Augmented audio
        """
        # Add noise
        if noise_level > 0:
            noise = np.random.randn(len(audio))
            audio = audio + noise_level * noise
            
        # Pitch shift
        if pitch_shift is not None:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.sample_rate,
                n_steps=pitch_shift
            )
            
        return audio
    
    def preprocess_for_model(
        self,
        file_path: str,
        apply_augmentation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Full preprocessing pipeline for model input
        
        Args:
            file_path: Path to audio file
            apply_augmentation: Whether to apply augmentations
            
        Returns:
            Dictionary of preprocessed features as torch tensors
        """
        # Load audio
        audio, _ = self.load_audio(file_path)
        
        # Apply augmentations if requested
        if apply_augmentation:
            audio = self.apply_augmentations(audio)
            
        # Extract features
        features = self.extract_features(audio)
        
        # Convert to torch tensors
        return {
            "mel_spectrogram": torch.from_numpy(features.mel_spectrogram).float(),
            "harmonic": torch.from_numpy(features.harmonic).float(),
            "percussive": torch.from_numpy(features.percussive).float(),
            "onset_envelope": torch.from_numpy(features.onset_envelope).float()
        } 