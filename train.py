import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np

from src.preprocessing.audio_processor import AudioProcessor
from src.models.transcriber import MusicTranscriber
from src.postprocessing.midi_generator import MIDIGenerator

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, processor: AudioProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.audio_files = []
        self.midi_files = []
        
        # Find all matching audio-MIDI pairs
        for audio_file in self.data_dir.glob("**/*.wav"):
            midi_file = audio_file.with_suffix(".mid")
            if midi_file.exists():
                self.audio_files.append(audio_file)
                self.midi_files.append(midi_file)
                
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        midi_path = self.midi_files[idx]
        
        # Load and preprocess audio
        features = self.processor.preprocess_for_model(
            str(audio_path),
            apply_augmentation=True
        )
        
        # Load MIDI ground truth
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Convert MIDI to target tensors
        target_onsets = torch.zeros(
            (1, features["mel_spectrogram"].shape[1])
        )
        target_pitches = {
            inst: torch.zeros(
                (1, features["mel_spectrogram"].shape[1], 61)  # C2 to C7
            )
            for inst in midi_data.instruments
        }
        target_instruments = torch.zeros(8)  # 8 supported instruments
        
        # Fill target tensors
        for instrument in midi_data.instruments:
            inst_name = instrument.name.lower()
            if inst_name in self.processor.instrument_programs:
                target_instruments[
                    self.processor.instrument_programs[inst_name]
                ] = 1.0
                
                for note in instrument.notes:
                    # Convert time to frame index
                    start_frame = int(
                        note.start * self.processor.sample_rate 
                        / self.processor.hop_length
                    )
                    end_frame = int(
                        note.end * self.processor.sample_rate 
                        / self.processor.hop_length
                    )
                    
                    # Mark onset
                    if start_frame < target_onsets.shape[1]:
                        target_onsets[0, start_frame] = 1.0
                    
                    # Mark pitch
                    pitch_idx = note.pitch - 36  # Convert from MIDI note number
                    if 0 <= pitch_idx < 61:  # C2 to C7
                        target_pitches[inst_name][
                            0, start_frame:end_frame, pitch_idx
                        ] = 1.0
        
        return {
            "features": features,
            "target_onsets": target_onsets,
            "target_pitches": target_pitches,
            "target_instruments": target_instruments
        }

class CompetitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        
    def forward(
        self,
        predictions: dict,
        targets: dict
    ) -> dict:
        # Onset detection loss
        onset_loss = self.bce(
            predictions["onset_times"],
            targets["target_onsets"]
        )
        
        # Pitch estimation loss
        pitch_loss = 0
        num_instruments = 0
        for inst_name, target in targets["target_pitches"].items():
            if inst_name in predictions["pitches"]:
                pitch_loss += self.bce(
                    predictions["pitches"][inst_name],
                    target
                )
                num_instruments += 1
        if num_instruments > 0:
            pitch_loss /= num_instruments
            
        # Instrument classification loss
        instrument_loss = self.bce(
            predictions["instruments"],
            targets["target_instruments"]
        )
        
        # Total loss with weighting
        total_loss = (
            onset_loss * 0.4 +  # Accurate timing is crucial
            pitch_loss * 0.4 +  # Pitch accuracy equally important
            instrument_loss * 0.2  # Instrument classification
        )
        
        return {
            "total": total_loss,
            "onset": onset_loss,
            "pitch": pitch_loss,
            "instrument": instrument_loss
        }

@hydra.main(config_path="config", config_name="train")
def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(
        project="music-transcription-competition",
        config=dict(cfg),
        name=cfg.experiment_name
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    processor = AudioProcessor()
    model = MusicTranscriber(
        pretrained_wav2vec=True,
        freeze_feature_extractor=False  # Fine-tune for competition
    ).to(device)
    criterion = CompetitionLoss()
    
    # Optimizer with learning rate scheduling
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Create datasets
    train_dataset = MusicDataset(
        os.path.join(cfg.data_dir, "train"),
        processor
    )
    val_dataset = MusicDataset(
        os.path.join(cfg.data_dir, "val"),
        processor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}") as pbar:
            for batch in pbar:
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor)
                    else {k2: v2.to(device) for k2, v2 in v.items()}
                    if isinstance(v, dict) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                predictions = model(batch["features"])
                losses = criterion(predictions, batch)
                
                # Backward pass
                optimizer.zero_grad()
                losses["total"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_losses.append(losses["total"].item())
                pbar.set_postfix({
                    "loss": np.mean(train_losses[-100:]),
                    "lr": scheduler.get_last_lr()[0]
                })
                
                # Log to wandb
                wandb.log({
                    "train/total_loss": losses["total"].item(),
                    "train/onset_loss": losses["onset"].item(),
                    "train/pitch_loss": losses["pitch"].item(),
                    "train/instrument_loss": losses["instrument"].item(),
                    "train/learning_rate": scheduler.get_last_lr()[0]
                })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor)
                    else {k2: v2.to(device) for k2, v2 in v.items()}
                    if isinstance(v, dict) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                predictions = model(batch["features"])
                losses = criterion(predictions, batch)
                val_losses.append(losses["total"].item())
                
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Log validation metrics
        wandb.log({
            "val/loss": val_loss,
            "epoch": epoch + 1
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(wandb.run.dir, "best_model.pt")
            )
            print("Saved new best model!")
            
    wandb.finish()

if __name__ == "__main__":
    train() 