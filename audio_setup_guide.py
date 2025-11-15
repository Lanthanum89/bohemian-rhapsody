#!/usr/bin/env python3
"""
Simple guide and helper for converting MIDI to MP3/WAV
"""

print("""
================================================================================
MIDI to MP3 Conversion Guide
================================================================================

Since you don't have FluidSynth or ffmpeg installed, here are the easiest options:

OPTION 1: Online Converter (Quickest)
--------------------------------------
1. Go to one of these websites:
   • https://www.zamzar.com/convert/midi-to-mp3/
   • https://www.online-convert.com/
   • https://convertio.co/midi-mp3/

2. Upload your file: "Queen - Bohemian Rhapsody.mid"

3. Convert to MP3 (or WAV)

4. Download the converted file

5. Save it as "bohemian_rhapsody.mp3" in this directory

6. Run: python bohemian_rhapsody_visualizer.py


OPTION 2: Install FluidSynth (Better Quality)
----------------------------------------------
1. Download from: https://github.com/FluidSynth/fluidsynth/releases
   
2. Install it (or use: choco install fluidsynth if you have Chocolatey)

3. Run: python convert_midi_to_wav.py

4. This will create bohemian_rhapsody.wav automatically


OPTION 3: Use a Different Audio File
-------------------------------------
If you have any audio version of Bohemian Rhapsody:
1. Name it "bohemian_rhapsody.mp3" or "bohemian_rhapsody.wav"
2. Place it in this directory
3. Run the visualizer

================================================================================
Current directory: """)

import os
print(os.getcwd())
print("\nLooking for audio files...")
for file in ["bohemian_rhapsody.mp3", "bohemian_rhapsody.wav", "Queen - Bohemian Rhapsody.mid"]:
    if os.path.exists(file):
        print(f"  ✓ Found: {file}")
    else:
        print(f"  ✗ Missing: {file}")

print("\n================================================================================")
