import numpy as np
import librosa
import torch
import torchaudio
from typing import Tuple, Dict, Optional, List, Union, Any
from dataclasses import dataclass
import random
from scipy import signal
import torch.nn.functional as F
import concurrent.futures
from functools import lru_cache
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import functools
import time

@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mel_spectrogram: np.ndarray
    harmonic: np.ndarray
    percussive: np.ndarray
    sample_rate: int
    onset_envelope: np.ndarray
    wav2vec_features: Optional[np.ndarray] = None
    noise_profile: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None

class AudioAugmenter:
    """Advanced audio augmentation pipeline"""
    def __init__(
        self,
        sample_rate: int,
        pitch_shift_range: Tuple[int, int] = (-2, 2),
        time_stretch_range: Tuple[float, float] = (0.95, 1.05),
        noise_snr_range: Tuple[float, float] = (20.0, 40.0)
    ):
        self.sample_rate = sample_rate
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.noise_snr_range = noise_snr_range
        
    def add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Add noise at specified SNR"""
        audio_rms = np.sqrt(np.mean(audio ** 2))
        noise = np.random.normal(0, 1, len(audio))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        
        snr_linear = 10 ** (snr_db / 20)
        noise_scaling = audio_rms / (noise_rms * snr_linear)
        return audio + noise * noise_scaling
    
    def apply_room_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simulate room acoustics"""
        # Simple convolution-based reverb
        reverb_length = int(self.sample_rate * 0.1)  # 100ms reverb
        reverb_ir = np.exp(-np.linspace(0, 5, reverb_length))
        return np.convolve(audio, reverb_ir, mode='same')
    
    def apply_augmentations(
        self,
        audio: np.ndarray,
        augment_prob: float = 0.5
    ) -> np.ndarray:
        """Apply random augmentations with probability"""
        if random.random() < augment_prob:
            # Pitch shift
            if random.random() < 0.7:
                n_steps = random.uniform(
                    self.pitch_shift_range[0],
                    self.pitch_shift_range[1]
                )
                audio = librosa.effects.pitch_shift(
                    audio,
                    sr=self.sample_rate,
                    n_steps=n_steps
                )
            
            # Time stretch
            if random.random() < 0.7:
                rate = random.uniform(
                    self.time_stretch_range[0],
                    self.time_stretch_range[1]
                )
                audio = librosa.effects.time_stretch(audio, rate=rate)
            
            # Add noise
            if random.random() < 0.5:
                snr = random.uniform(
                    self.noise_snr_range[0],
                    self.noise_snr_range[1]
                )
                audio = self.add_noise(audio, snr)
            
            # Add reverb
            if random.random() < 0.3:
                audio = self.apply_room_reverb(audio)
                
        return audio

class AudioProcessor:
    """Audio processing and feature extraction"""
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: int = 20,
        f_max: int = 8000,
        use_cqt: bool = False,
        parallel_extraction: bool = False,
        num_workers: int = 4
    ):
        """
        Audio processor class for feature extraction
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency
            use_cqt: Whether to use constant-Q transform
            parallel_extraction: Whether to extract features in parallel
            num_workers: Number of workers for parallel extraction
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.use_cqt = use_cqt
        self.parallel_extraction = parallel_extraction
        self.num_workers = num_workers
        
        # Precompute mel filterbank for efficiency
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        # Cache for frequently used features
        self._feature_cache = {}
        self._max_cache_size = 50  # Maximum number of files to cache
        
        # Cache window
        self.window = torch.hann_window(n_fft)
        
        # Define instrument-specific frequency ranges
        self.instrument_ranges = {
            'piano': (27.5, 4186.0),
            'violin': (196.0, 3136.0),
            'cello': (65.4, 988.0),
            'flute': (261.6, 2093.0),
            'clarinet': (146.8, 1976.0),
            'oboe': (246.9, 1976.0),
            'bassoon': (58.3, 587.3),
            'trumpet': (164.8, 1047.0),
            'trombone': (82.4, 698.5),
            'horn': (87.3, 880.0),
            'tuba': (43.7, 349.2),
            'guitar': (82.4, 1319.0),
            'bass': (41.2, 392.0)
        }
        
        # Initialize thread pool for parallel processing
        if parallel_extraction:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    @lru_cache(maxsize=32)
    def _get_cqt_kernel(self, n_bins: int = 84, bins_per_octave: int = 12) -> torch.Tensor:
        """Get cached CQT kernel for efficient constant-Q transform"""
        # Generate CQT kernels
        cqt_kernels, _ = librosa.filters.constant_q(
            sr=self.sample_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=librosa.note_to_hz('C1')
        )
        
        # Convert to tensor
        cqt_kernels = torch.from_numpy(cqt_kernels).float()
        return cqt_kernels
    
    def compute_cqt(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute constant-Q transform (better for music)"""
        # Get CQT kernels
        cqt_kernels = self._get_cqt_kernel()
        
        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        # Convert to magnitude
        stft_mag = torch.abs(stft)
        
        # Apply CQT kernels
        cqt = torch.matmul(cqt_kernels, stft_mag)
        
        # Convert to log scale
        cqt = torch.log(cqt + 1e-9)
        
        return cqt
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to the range [-1, 1]
        """
        if audio.size == 0:
            return audio
            
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Apply peak normalization
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
            
        return audio
        
    def load_audio(self, audio_file: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file
        """
        try:
            if isinstance(audio_file, (str, os.PathLike)):
                # Determine optimal duration to load based on file size
                file_size = os.path.getsize(audio_file)
                if file_size > 50 * 1024 * 1024:  # If larger than 50MB
                    duration = 30.0  # Only load first 30 seconds
                else:
                    duration = None
                    
                audio, sr = librosa.load(
                    audio_file, 
                    sr=self.sample_rate, 
                    mono=True,
                    duration=duration
                )
            else:
                # Assume it's already loaded audio data
                audio = audio_file
                sr = self.sample_rate
                
            # Normalize audio
            audio = self.normalize_audio(audio)
            
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Return empty audio and sample rate
            return np.zeros(0), self.sample_rate
    
    def compute_stft(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute STFT for audio
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
            
        # Perform STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to power spectrogram
        spec = np.abs(stft) ** 2
        
        return spec
        
    def compute_mel_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Efficiently compute mel spectrogram
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Compute power spectrogram
        spec = self.compute_stft(audio)
        
        # Apply mel filterbank (use precomputed mel basis)
        mel_spec = np.dot(self.mel_basis, spec)
        
        # Convert to log scale (add small constant to avoid log(0))
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize to [-1, 1] for better model convergence
        mel_spec = (mel_spec / 40.0) - 1.0  # Normalize to [-1, 1] range
        
        return torch.from_numpy(mel_spec.astype(np.float32))
    
    def compute_harmonic_percussive(self, audio_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute harmonic and percussive components using median filtering"""
        # Compute STFT
        D = librosa.stft(audio_np, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Perform harmonic-percussive separation
        H, P = librosa.decompose.hpss(D)
        
        # Convert to tensors
        harmonic = torch.from_numpy(np.abs(H)).float()
        percussive = torch.from_numpy(np.abs(P)).float()
        
        return harmonic, percussive
    
    def compute_onset_strength(self, percussive: torch.Tensor) -> torch.Tensor:
        """Compute onset strength envelope from percussive component"""
        onset_env = librosa.onset.onset_strength(
            S=percussive.numpy(),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return torch.from_numpy(onset_env).float()
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Process a chunk of audio and extract features
        """
        # Normalize audio
        audio_chunk = self.normalize_audio(audio_chunk)
        
        # Extract features
        features = {}
        
        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio_chunk)
        features["mel_spectrogram"] = mel_spec
        
        # Compute CQT if enabled
        if self.use_cqt:
            cqt = self.compute_cqt(torch.from_numpy(audio_chunk).float())
            features["cqt"] = cqt
            
        # Include raw audio
        features["audio"] = torch.from_numpy(audio_chunk.astype(np.float32))
        
        return features
    
    def process_audio_file(self, audio_file: str) -> Dict[str, torch.Tensor]:
        """
        Process audio file and extract features
        """
        # Check cache first
        if audio_file in self._feature_cache:
            return self._feature_cache[audio_file]
            
        # Load audio
        audio, sr = self.load_audio(audio_file)
        
        # Early return for empty audio
        if audio.size == 0:
            return {
                "mel_spectrogram": torch.zeros((self.n_mels, 1), dtype=torch.float32),
                "audio": torch.zeros(1, dtype=torch.float32)
            }
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Process chunk directly
        features = self.process_audio_chunk(audio)
        
        # Add to cache (with LRU management)
        if len(self._feature_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._feature_cache))
            del self._feature_cache[oldest_key]
            
        # Add to cache
        self._feature_cache[audio_file] = features
        
        return features
    
    def process_audio_chunks_parallel(self, audio_chunks: List[np.ndarray]) -> List[Dict[str, torch.Tensor]]:
        """
        Process multiple audio chunks in parallel
        """
        if not self.parallel_extraction or len(audio_chunks) <= 1:
            # Process sequentially if parallel disabled or only one chunk
            return [self.process_audio_chunk(chunk) for chunk in audio_chunks]
            
        # Process in parallel using thread pool
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            features = list(executor.map(self.process_audio_chunk, audio_chunks))
            
        return features
    
    def extract_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from audio tensor
        Args:
            audio: Audio tensor of shape [batch_size, samples]
        Returns:
            Dictionary containing audio features
        """
        # Check cache using hash of tensor
        tensor_hash = str(hash(audio.cpu().numpy().tobytes()))
        cache_key = f"features_tensor_{tensor_hash}"
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
            
        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)  # [batch_size, n_mels, time]
        
        # Add channel dimension for CNN input
        mel_spec = mel_spec.unsqueeze(1)  # [batch_size, 1, n_mels, time]
        
        # Feature extraction can be split into parallel tasks
        if self.parallel_extraction:
            # Move audio to CPU for librosa processing
            audio_np = audio.squeeze().cpu().numpy()
            
            # Submit tasks
            harmonic_percussive_future = self.executor.submit(
                self.compute_harmonic_percussive, audio_np
            )
            
            # Get results
            harmonic, percussive = harmonic_percussive_future.result()
            
            # Compute onset envelope from percussive component
            onset_env = self.compute_onset_strength(percussive)
            
            # Normalize components
            harmonic = (harmonic - harmonic.mean()) / (harmonic.std() + 1e-8)
            percussive = (percussive - percussive.mean()) / (percussive.std() + 1e-8)
        else:
            # Sequential processing
            # Move audio to CPU for librosa processing
            audio_np = audio.squeeze().cpu().numpy()
            
            # Compute STFT for harmonic-percussive separation
            D = librosa.stft(audio_np, n_fft=self.n_fft, hop_length=self.hop_length)
            H, P = librosa.decompose.hpss(D)
            
            # Convert to tensors and normalize
            harmonic = torch.from_numpy(np.abs(H)).float()
            percussive = torch.from_numpy(np.abs(P)).float()
            
            # Normalize
            harmonic = (harmonic - harmonic.mean()) / (harmonic.std() + 1e-8)
            percussive = (percussive - percussive.mean()) / (percussive.std() + 1e-8)
            
            # Compute onset envelope from percussive component
            onset_env = librosa.onset.onset_strength(
                S=percussive.numpy(),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            onset_env = torch.from_numpy(onset_env).float()
        
        # Create features dictionary
        features = {
            "mel_spectrogram": mel_spec,
            "harmonic": harmonic,
            "percussive": percussive,
            "onset_envelope": onset_env
        }
        
        # Add CQT if enabled
        if self.use_cqt:
            features["cqt"] = self.compute_cqt(audio)
        
        # Cache features
        self._feature_cache[cache_key] = features
        
        return features
        
    def split_audio_into_chunks(
        self, 
        audio: torch.Tensor, 
        chunk_size: int = 4096, 
        overlap: float = 0.25
    ) -> List[torch.Tensor]:
        """Split audio into overlapping chunks for efficient processing
        
        Args:
            audio: Audio tensor [samples]
            chunk_size: Size of each chunk in samples
            overlap: Overlap between chunks (0-1)
            
        Returns:
            List of audio chunks
        """
        # Calculate overlap in samples
        overlap_samples = int(chunk_size * overlap)
        hop_size = chunk_size - overlap_samples
        
        # Split audio into chunks
        chunks = []
        for i in range(0, len(audio), hop_size):
            chunk = audio[i:i+chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                padding = chunk_size - len(chunk)
                chunk = F.pad(chunk, (0, padding))
                
            chunks.append(chunk)
            
        return chunks
    
    def batch_process_chunks(self, chunks: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Process multiple chunks in parallel
        
        Args:
            chunks: List of audio chunks
            
        Returns:
            List of features for each chunk
        """
        if self.parallel_extraction:
            # Submit processing tasks
            futures = []
            for chunk in chunks:
                futures.append(
                    self.executor.submit(
                        self.process_audio_chunk,
                        chunk.numpy()
                    )
                )
                
            # Get results
            return [future.result() for future in futures]
        else:
            # Sequential processing
            return [
                self.process_audio_chunk(chunk.numpy())
                for chunk in chunks
            ]
    
    def clear_cache(self):
        """Clear the feature cache to free memory"""
        self._feature_cache.clear()
        
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown()

    def preprocess_for_model(
        self,
        file_path: str,
        apply_augmentation: bool = False
    ) -> AudioFeatures:
        """Load and preprocess audio file for model input"""
        try:
            # Load and preprocess audio
            audio = self.load_audio(file_path)
            
            # Extract features
            features = self.extract_features(audio)
            
            return features
            
        except Exception as e:
            print(f"Error preprocessing audio file {file_path}: {str(e)}")
            return None 