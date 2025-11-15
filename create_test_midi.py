#!/usr/bin/env python3
"""
Quick test script to verify the visualiser works with a simple generated MIDI file
"""

import mido
import random


def create_test_midi():
    """Create a simple test MIDI file with dramatic changes (Bohemian Rhapsody style)"""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo
    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM
    
    # Section 1: Lower notes (C major scale ascending)
    base_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    for _ in range(4):
        for note in base_notes:
            velocity = random.randint(60, 90)
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=240))
    
    # Build up to key change
    track.append(mido.Message('note_on', note=67, velocity=100, time=0))
    track.append(mido.Message('note_off', note=67, velocity=0, time=120))
    track.append(mido.Message('note_on', note=69, velocity=110, time=0))
    track.append(mido.Message('note_off', note=69, velocity=0, time=120))
    track.append(mido.Message('note_on', note=71, velocity=120, time=0))
    track.append(mido.Message('note_off', note=71, velocity=0, time=120))
    
    # KEY CHANGE! - Dramatic high note
    track.append(mido.Message('note_on', note=76, velocity=127, time=0))
    track.append(mido.Message('note_off', note=76, velocity=0, time=480))
    
    # Section 2: Higher notes (one step up)
    high_notes = [62, 64, 66, 67, 69, 71, 73, 74]
    for _ in range(4):
        for note in high_notes:
            velocity = random.randint(70, 100)
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=240))
    
    # Dramatic ending
    for note in [74, 76, 79, 81]:
        track.append(mido.Message('note_on', note=note, velocity=127, time=0))
        track.append(mido.Message('note_off', note=note, velocity=0, time=360))
    
    # Save
    mid.save('test_bohemian_rhapsody.mid')
    print("✓ Created test_bohemian_rhapsody.mid")
    print(f"  Total length: ~{sum(msg.time for msg in track) / 480:.1f} seconds")
    print(f"  Total notes: {sum(1 for msg in track if msg.type == 'note_on')}")


if __name__ == "__main__":
    create_test_midi()
    print("\nNow run:")
    print("  python bohemian_rhapsody_visualizer.py test_bohemian_rhapsody.mid")
