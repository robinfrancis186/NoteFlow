#!/usr/bin/env python
import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import pretty_midi

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.competition.transcriber import CompetitionTranscriber, CompetitionMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Run competition evaluation")
    parser.add_argument("--data-dir", required=True, help="Path to test data directory")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    return parser.parse_args()

def setup_logging(output_dir):
    """Set up logging to file and console"""
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamped log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"competition_run_{timestamp}.log"
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"Logs will be saved to: {log_file.parent}")
    return str(log_dir)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    log_dir = setup_logging(output_dir)
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        device = "cpu"
    
    # Initialize transcriber
    transcriber = CompetitionTranscriber(device=device)
    
    # Find all WAV files in the data directory
    wav_files = list(data_dir.glob("**/*.wav"))
    print(f"Found {len(wav_files)} test files")
    
    # Process each file
    start_time = time.time()
    successful_files = 0
    metrics = {
        "onset_accuracy": 0.0,
        "pitch_accuracy": 0.0,
        "instrument_accuracy": 0.0
    }
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        try:
            # Process the file
            file_metrics, predictions = transcriber._process_single(str(wav_file))
            
            # Check if we have predictions
            if predictions and len(predictions.get("onset_times", [])) > 0:
                # Create MIDI file
                midi_data = transcriber._predictions_to_midi(predictions)
                
                # Create output file path preserving directory structure
                rel_path = wav_file.relative_to(data_dir)
                output_file = output_dir / rel_path.with_suffix(".mid")
                output_file.parent.mkdir(exist_ok=True, parents=True)
                
                # Save MIDI file
                midi_data.write(str(output_file))
                
                # Update metrics
                for key in metrics:
                    metrics[key] += file_metrics[key]
                
                successful_files += 1
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
    
    # Calculate total time and average metrics
    total_time = time.time() - start_time
    avg_time = total_time / max(1, len(wav_files))
    
    for key in metrics:
        metrics[key] = metrics[key] / max(1, successful_files)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Files processed: {successful_files}")
    print(f"Average time per file: {avg_time:.2f}s")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    
    # Save metrics to JSON file
    metrics_file = Path(log_dir) / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "total_files": len(wav_files),
            "successful_files": successful_files,
            "total_time": total_time,
            "average_time": avg_time,
            "metrics": metrics
        }, f, indent=2)

if __name__ == "__main__":
    main() 