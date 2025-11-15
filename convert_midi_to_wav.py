#!/usr/bin/env python3
"""
Convert MIDI file to WAV audio using FluidSynth
"""
from midi2audio import FluidSynth

def convert_midi_to_audio():
    """Convert the MIDI file to WAV format"""
    try:
        # Initialize FluidSynth
        fs = FluidSynth()
        
        # Convert MIDI to WAV
        print("Converting MIDI to audio...")
        fs.midi_to_audio('Queen - Bohemian Rhapsody.mid', 'bohemian_rhapsody.wav')
        print("✓ Successfully created bohemian_rhapsody.wav")
        print("\nYou can now run the visualiser with audio!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\nThis requires FluidSynth to be installed on your system.")
        print("Please follow the instructions in setup_audio.md")

if __name__ == "__main__":
    convert_midi_to_audio()
