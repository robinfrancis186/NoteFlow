import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch

@dataclass
class Note:
    """Container for MIDI note information"""
    pitch: int
    start_time: float
    end_time: float
    velocity: int = 100
    instrument: str = "piano"

class MIDIGenerator:
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_note_length: float = 0.1,
        velocity_scale: float = 100.0
    ):
        """Initialize MIDI generator
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length used in feature extraction
            min_note_length: Minimum note duration in seconds
            velocity_scale: Scaling factor for note velocities
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_note_length = min_note_length
        self.velocity_scale = velocity_scale
        
        # Competition-specific settings
        self.min_tempo = 60.0
        self.max_tempo = 90.0
        self.min_pitch = 36  # C2
        self.max_pitch = 96  # C7
        self.supported_time_sigs = [(3, 4), (4, 4), (6, 8)]
        
        # Instrument program numbers
        self.instrument_programs = {
            'piano': 0,
            'violin': 40,
            'cello': 42,
            'flute': 73,
            'bassoon': 70,
            'trombone': 57,
            'oboe': 68,
            'viola': 41
        }
        
    def frames_to_time(self, frames: int) -> float:
        """Convert frame index to time in seconds"""
        return frames * self.hop_length / self.sample_rate
    
    def detect_time_signature(
        self,
        onsets: np.ndarray,
        tempo: float
    ) -> Tuple[int, int]:
        """Detect time signature from onset pattern
        
        Args:
            onsets: Binary onset detection
            tempo: Tempo in BPM
            
        Returns:
            Tuple of (numerator, denominator)
        """
        # Convert onsets to beat-aligned grid
        beat_duration = 60.0 / tempo
        frames_per_beat = int(beat_duration * self.sample_rate / self.hop_length)
        
        # Count onsets in different beat groupings
        beat_onsets = np.sum(onsets.reshape(-1, frames_per_beat), axis=1)
        
        # Analyze patterns for 3/4, 4/4, and 6/8
        patterns = {
            (3, 4): np.sum(beat_onsets.reshape(-1, 3), axis=0),
            (4, 4): np.sum(beat_onsets.reshape(-1, 4), axis=0),
            (6, 8): np.sum(beat_onsets.reshape(-1, 6), axis=0)
        }
        
        # Choose time signature with strongest beat pattern
        best_score = -1
        best_time_sig = (4, 4)  # Default
        
        for time_sig, pattern in patterns.items():
            # Score based on first beat emphasis and overall pattern clarity
            first_beat_strength = pattern[0] / (np.mean(pattern[1:]) + 1e-8)
            pattern_clarity = np.std(pattern)
            score = first_beat_strength * pattern_clarity
            
            if score > best_score:
                best_score = score
                best_time_sig = time_sig
                
        return best_time_sig
    
    def detect_note_segments(
        self,
        onsets: torch.Tensor,
        pitches: Dict[str, torch.Tensor],
        instruments: List[str],
        min_frames: int = 2
    ) -> List[Note]:
        """Convert frame-wise predictions to note segments
        
        Args:
            onsets: Binary onset detection
            pitches: Dictionary of pitch activations per instrument
            instruments: List of active instruments
            min_frames: Minimum number of frames for a valid note
            
        Returns:
            List of Note objects
        """
        notes = []
        current_notes = {}  # (instrument, pitch) -> start_frame
        
        # Convert to numpy for easier processing
        onsets = onsets.cpu().numpy()
        pitch_arrays = {
            inst: pitch.cpu().numpy()
            for inst, pitch in pitches.items()
        }
        
        for frame in range(len(onsets[0])):
            # Process note onsets
            if onsets[0, frame] > 0:
                for inst in instruments:
                    active_pitches = np.where(pitch_arrays[inst][0, frame] > 0)[0]
                    for pitch in active_pitches:
                        # Validate pitch range
                        midi_pitch = pitch + 36  # Convert to MIDI note number
                        if self.min_pitch <= midi_pitch <= self.max_pitch:
                            key = (inst, pitch)
                            if key not in current_notes:
                                current_notes[key] = frame
                        
            # Process note offsets
            for key in list(current_notes.keys()):
                inst, pitch = key
                if frame - current_notes[key] >= min_frames:
                    if pitch_arrays[inst][0, frame, pitch] == 0:
                        # Note ended
                        start_frame = current_notes[key]
                        if frame - start_frame >= min_frames:
                            notes.append(Note(
                                pitch=int(pitch) + 36,  # Convert to MIDI note number
                                start_time=self.frames_to_time(start_frame),
                                end_time=self.frames_to_time(frame),
                                velocity=100,
                                instrument=inst
                            ))
                        del current_notes[key]
                        
        # Handle notes that are still active at the end
        final_frame = len(onsets[0]) - 1
        for (inst, pitch), start_frame in current_notes.items():
            if final_frame - start_frame >= min_frames:
                notes.append(Note(
                    pitch=int(pitch) + 36,
                    start_time=self.frames_to_time(start_frame),
                    end_time=self.frames_to_time(final_frame),
                    velocity=100,
                    instrument=inst
                ))
                
        return notes
    
    def apply_timing_correction(
        self,
        notes: List[Note],
        tempo: float,
        time_signature: Tuple[int, int]
    ) -> List[Note]:
        """Quantize note timings to a musical grid
        
        Args:
            notes: List of detected notes
            tempo: Tempo in BPM
            time_signature: Tuple of (numerator, denominator)
            
        Returns:
            List of quantized notes
        """
        # Ensure tempo is within competition bounds
        tempo = max(self.min_tempo, min(self.max_tempo, tempo))
        
        # Calculate grid resolution based on time signature
        beat_duration = 60.0 / tempo
        if time_signature[1] == 8:
            grid_resolution = beat_duration / 6  # For compound meters
        else:
            grid_resolution = beat_duration / 4  # For simple meters
        
        quantized_notes = []
        for note in notes:
            # Quantize start and end times
            start_grid = round(note.start_time / grid_resolution) * grid_resolution
            end_grid = round(note.end_time / grid_resolution) * grid_resolution
            
            # Ensure minimum note length
            if end_grid - start_grid >= self.min_note_length:
                quantized_notes.append(Note(
                    pitch=note.pitch,
                    start_time=start_grid,
                    end_time=end_grid,
                    velocity=note.velocity,
                    instrument=note.instrument
                ))
                
        return quantized_notes
    
    def create_midi(
        self,
        notes: List[Note],
        tempo: float,
        time_signature: Tuple[int, int]
    ) -> pretty_midi.PrettyMIDI:
        """Create a MIDI file from note list
        
        Args:
            notes: List of notes to include
            tempo: Tempo in BPM
            time_signature: Tuple of (numerator, denominator)
            
        Returns:
            PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # Add time signature
        ts = pretty_midi.TimeSignature(
            numerator=time_signature[0],
            denominator=time_signature[1],
            time=0
        )
        midi.time_signatures = [ts]
        
        # Group notes by instrument
        instruments = {}
        for note in notes:
            if note.instrument not in instruments:
                instruments[note.instrument] = []
            instruments[note.instrument].append(note)
            
        # Create instrument programs
        for inst_name, inst_notes in instruments.items():
            instrument = pretty_midi.Instrument(
                program=self.instrument_programs[inst_name],
                is_drum=False,
                name=inst_name
            )
            
            # Add notes to instrument
            for note in inst_notes:
                midi_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start_time,
                    end=note.end_time
                )
                instrument.notes.append(midi_note)
                
            midi.instruments.append(instrument)
            
        return midi
    
    def predictions_to_midi(
        self,
        model_output: Dict[str, torch.Tensor],
        tempo: Optional[float] = None
    ) -> pretty_midi.PrettyMIDI:
        """Convert model predictions to MIDI file
        
        Args:
            model_output: Dictionary containing onset_times, pitches, and active_instruments
            tempo: Optional tempo in BPM (will be detected if not provided)
            
        Returns:
            PrettyMIDI object
        """
        onsets = model_output["onset_times"]
        
        # Detect tempo if not provided
        if tempo is None:
            # Simple tempo detection based on onset spacing
            onset_times = np.where(onsets.cpu().numpy()[0] > 0)[0]
            if len(onset_times) > 1:
                median_spacing = np.median(np.diff(onset_times))
                tempo = 60.0 / (median_spacing * self.hop_length / self.sample_rate)
                tempo = max(self.min_tempo, min(self.max_tempo, tempo))
            else:
                tempo = 120.0
        
        # Detect time signature
        time_signature = self.detect_time_signature(
            onsets.cpu().numpy()[0],
            tempo
        )
        
        # Detect note segments
        notes = self.detect_note_segments(
            onsets,
            model_output["pitches"],
            model_output["active_instruments"]
        )
        
        # Apply timing correction
        notes = self.apply_timing_correction(notes, tempo, time_signature)
        
        # Create MIDI file
        return self.create_midi(notes, tempo, time_signature) 