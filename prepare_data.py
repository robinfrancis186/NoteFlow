import os
import shutil
import argparse
from pathlib import Path
import pretty_midi
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import yaml

def load_config(config_path: str = "config/train.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def validate_audio(
    audio_path: str,
    sample_rate: int = 44100,
    min_duration: float = 5.0,
    max_duration: float = 20.0
) -> tuple[bool, str]:
    """Validate audio file meets competition requirements."""
    try:
        duration = librosa.get_duration(path=audio_path)
        if not (min_duration <= duration <= max_duration):
            return False, f"Duration {duration:.2f}s outside range [{min_duration}, {max_duration}]"
            
        y, sr = librosa.load(audio_path, sr=None)
        if sr != sample_rate:
            return False, f"Sample rate {sr} Hz does not match required {sample_rate} Hz"
            
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def validate_midi(
    midi_path: str,
    config: dict
) -> tuple[bool, str]:
    """Validate MIDI file meets competition requirements."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Check tempo range
        if not (config["competition"]["min_tempo"] <= midi_data.get_tempo_changes()[1][0] <= config["competition"]["max_tempo"]):
            return False, "Tempo outside competition range"
            
        # Check time signature
        time_sigs = midi_data.time_signature_changes
        if not time_sigs:
            return False, "No time signature found"
            
        valid_time_sigs = config["competition"]["supported_time_sigs"]
        if not any([ts.numerator == sig[0] and ts.denominator == sig[1] 
                   for ts in time_sigs for sig in valid_time_sigs]):
            return False, "Unsupported time signature"
            
        # Check number of instruments
        if len(midi_data.instruments) > config["competition"]["max_instruments"]:
            return False, "Too many instruments"
            
        # Check pitch range
        min_pitch = config["competition"]["min_pitch"]
        max_pitch = config["competition"]["max_pitch"]
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                pitches = [note.pitch for note in instrument.notes]
                if pitches and (min(pitches) < min_pitch or max(pitches) > max_pitch):
                    return False, f"Pitch range {min(pitches)}-{max(pitches)} outside allowed range {min_pitch}-{max_pitch}"
                    
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def process_file_pair(args) -> dict:
    """Process and validate an audio-MIDI pair."""
    audio_path, midi_path, output_dir, config = args
    
    result = {
        "audio": str(audio_path),
        "midi": str(midi_path),
        "status": "failed",
        "message": ""
    }
    
    # Validate files
    audio_valid, audio_msg = validate_audio(
        audio_path,
        config["audio"]["sample_rate"]
    )
    if not audio_valid:
        result["message"] = f"Audio validation failed: {audio_msg}"
        return result
        
    midi_valid, midi_msg = validate_midi(midi_path, config)
    if not midi_valid:
        result["message"] = f"MIDI validation failed: {midi_msg}"
        return result
        
    try:
        # Create output directories
        rel_path = os.path.relpath(
            audio_path,
            start=os.path.commonpath([audio_path, output_dir])
        )
        output_audio = Path(output_dir) / rel_path
        output_midi = output_audio.with_suffix(".mid")
        
        os.makedirs(output_audio.parent, exist_ok=True)
        
        # Copy files
        shutil.copy2(audio_path, output_audio)
        shutil.copy2(midi_path, output_midi)
        
        result["status"] = "success"
        result["message"] = "Files processed successfully"
        
    except Exception as e:
        result["message"] = f"Processing failed: {str(e)}"
        
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate training data for music transcription"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input audio-MIDI pairs"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to store processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all audio-MIDI pairs
    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.rglob("*.wav"))
    file_pairs = []
    
    for audio_path in audio_files:
        midi_path = audio_path.with_suffix(".mid")
        if midi_path.exists():
            file_pairs.append((
                str(audio_path),
                str(midi_path),
                args.output_dir,
                config
            ))
    
    print(f"Found {len(file_pairs)} audio-MIDI pairs")
    
    # Process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for result in tqdm(
            executor.map(process_file_pair, file_pairs),
            total=len(file_pairs),
            desc="Processing files"
        ):
            results.append(result)
    
    # Save processing report
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    report = {
        "total_files": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "failed_files": failed
    }
    
    report_path = Path(args.output_dir) / "processing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"See {report_path} for details")

if __name__ == "__main__":
    main() 