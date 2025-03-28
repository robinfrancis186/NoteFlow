import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from transformers import Wav2Vec2Model

SUPPORTED_INSTRUMENTS = {
    'piano': 0,
    'violin': 1,
    'cello': 2,
    'flute': 3,
    'bassoon': 4,
    'trombone': 5,
    'oboe': 6,
    'viola': 7
}

class OnsetDetector(nn.Module):
    """CNN-RNN hybrid for onset detection"""
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        
        # CNN layers with increased capacity
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Bidirectional LSTM with increased capacity
        self.lstm = nn.LSTM(
            input_size=256 * 16,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Reshape for LSTM
        batch_size, channels, time_steps, mel_bands = x.shape
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, time_steps, channels * mel_bands)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        return torch.sigmoid(self.fc(x))

class PitchEstimator(nn.Module):
    """Transformer-based pitch estimation"""
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,  # Increased layers
        dim_feedforward: int = 2048,
        num_pitches: int = 61  # C2 to C7 = 61 notes
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward * 2,  # Increased capacity
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(dim_feedforward, num_pitches)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return torch.sigmoid(self.output_projection(x))

class InstrumentClassifier(nn.Module):
    """Multi-instrument classifier"""
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        num_instruments: int = len(SUPPORTED_INSTRUMENTS)
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_instruments)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)

class MusicTranscriber(nn.Module):
    """Complete music transcription model optimized for competition"""
    def __init__(
        self,
        pretrained_wav2vec: bool = True,
        freeze_feature_extractor: bool = True
    ):
        super().__init__()
        
        # Load pretrained wav2vec model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_feature_extractor:
            self.wav2vec.feature_extractor._freeze_parameters()
            
        # Onset detection
        self.onset_detector = OnsetDetector()
        
        # Pitch estimation (one per instrument)
        self.pitch_estimators = nn.ModuleDict({
            name: PitchEstimator() 
            for name in SUPPORTED_INSTRUMENTS.keys()
        })
        
        # Instrument classification
        self.instrument_classifier = InstrumentClassifier()
        
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Extract wav2vec features
        wav2vec_features = self.wav2vec(
            features["harmonic"],
            output_hidden_states=True
        ).hidden_states[-1]
        
        # Detect onsets
        mel_spec = features["mel_spectrogram"].unsqueeze(1)
        onsets = self.onset_detector(mel_spec)
        
        # Classify instruments
        instrument_probs = self.instrument_classifier(mel_spec)
        
        # Estimate pitches for each instrument
        all_pitches = {}
        for inst_name in SUPPORTED_INSTRUMENTS:
            pitches = self.pitch_estimators[inst_name](wav2vec_features)
            all_pitches[inst_name] = pitches
        
        return {
            "onset_times": onsets,
            "pitches": all_pitches,
            "instruments": instrument_probs
        }
    
    def transcribe(
        self,
        features: Dict[str, torch.Tensor],
        onset_threshold: float = 0.5,
        pitch_threshold: float = 0.5,
        instrument_threshold: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """Transcribe audio to MIDI-like representation
        
        Args:
            features: Dictionary of preprocessed audio features
            onset_threshold: Threshold for onset detection
            pitch_threshold: Threshold for pitch detection
            instrument_threshold: Threshold for instrument detection
            
        Returns:
            Dictionary containing onset times, pitches, and instruments
        """
        predictions = self.forward(features)
        
        # Threshold predictions
        onset_pred = (predictions["onset_times"] > onset_threshold).float()
        instrument_pred = (predictions["instruments"] > instrument_threshold).float()
        
        # Get active instruments
        active_instruments = []
        for i, is_active in enumerate(instrument_pred[0]):
            if is_active:
                inst_name = list(SUPPORTED_INSTRUMENTS.keys())[i]
                active_instruments.append(inst_name)
        
        # Limit to top 3 instruments as per competition rules
        if len(active_instruments) > 3:
            instrument_probs = predictions["instruments"][0]
            top_3_indices = torch.topk(instrument_probs, k=3).indices
            active_instruments = [
                list(SUPPORTED_INSTRUMENTS.keys())[i] 
                for i in top_3_indices
            ]
        
        # Get pitch predictions for active instruments
        pitch_preds = {}
        for inst_name in active_instruments:
            pitch_preds[inst_name] = (
                predictions["pitches"][inst_name] > pitch_threshold
            ).float()
        
        return {
            "onset_times": onset_pred,
            "pitches": pitch_preds,
            "active_instruments": active_instruments
        } 