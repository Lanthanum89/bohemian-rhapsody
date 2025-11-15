# 👑 Bohemian Rhapsody Music Visualiser

A MIDI-driven music visualiser inspired by Queen's "Bohemian Rhapsody", featuring geometric particle bursts, sparkle effects for dramatic moments, and a completely pointless but fun ML model that predicts whether the next note will be high or low!

## Features

✨ **Reactive Visualisation**

- Green and purple geometric bursts (circles, squares, triangles)
- Particle intensity scales with note velocity
- Note pitch determines vertical position

✨ **Special Effects**

- Sparkle explosions at high-intensity moments (simulating key changes)
- Smooth particle physics with gravity and rotation

✨ **ML Prediction** (The Fun Part!)

- Random Forest classifier trained on the MIDI sequence
- Predicts if the next note will be HIGH or LOW
- Displays prediction confidence in real-time
- Completely unnecessary but delightful!

## Installation

```bash
pip install -r requirements.txt
```

## Getting Files

### MIDI File (Required for visualisation)

You'll need a MIDI version of "Bohemian Rhapsody". You can find MIDI files on:

- [MuseScore](https://musescore.com) - Search for "Bohemian Rhapsody Queen"
- [MidiWorld](https://www.midiworld.com)
- [BitMidi](https://bitmidi.com)

Save the file as `Queen - Bohemian Rhapsody.mid` in this directory.

### Audio File (Optional but recommended for sound)

For audio playback, add an MP3 file:

- Download or convert "Bohemian Rhapsody" to MP3 format
- Save as `bohemian_rhapsody.mp3` in this directory
- The visualiser will automatically detect and play it

**Note**: MIDI audio playback on Windows often doesn't work without additional drivers. Using an MP3 file is the easiest way to hear the music.

## Usage

```bash
# With default filename (Queen - Bohemian Rhapsody.mid)
python bohemian_rhapsody_visualizer.py

# With custom MIDI file
python bohemian_rhapsody_visualizer.py path/to/your/midi/file.mid
```

## Controls

- **SPACE**: Pause/Resume
- **ESC**: Exit

## How It Works

### MIDI Parsing

Uses `mido` library to parse MIDI files and extract:

- Note timing (converted to seconds)
- Note pitch (0-127)
- Note velocity (intensity)
- Tempo changes

### Visualisation

- **Green particles** = Higher notes (≥ middle C)
- **Purple particles** = Lower notes (< middle C)
- Particle count and size scale with velocity
- Three geometric shapes: circles, squares, triangles
- Physics simulation with gravity and rotation

### Sparkle Effects

Triggered when:

- Note velocity exceeds 80% threshold
- At least 5 seconds since last sparkle
- Creates white/gold/pink particle explosions

### ML Note Prediction

1. **Training**: Analyzes entire MIDI sequence
2. **Features**: Last 5 notes + their velocities (10 features)
3. **Label**: Next note is HIGH (above median) or LOW (below median)
4. **Model**: Random Forest with 50 trees
5. **Prediction**: Updates every 3 notes during playback

The ML model typically achieves 60-70% accuracy, which is hilariously better than random chance for something this silly!

## Code Structure

```
bohemian_rhapsody_visualizer.py
├── NotePredictor       # ML model for note prediction
├── Particle            # Individual geometric particle
├── Sparkle             # Dramatic moment sparkle effect
└── BohemianRhapsodyVisualizer  # Main visualiser class
    ├── MIDI parsing
    ├── Particle management
    ├── Effect triggering
    └── Rendering loop
```

## Technical Details

- **Frame rate**: 60 FPS
- **Resolution**: 1200x800
- **Particle lifetime**: ~0.7 seconds
- **ML sequence length**: 5 notes
- **Physics**: Simple gravity (200 px/s²)

## Why the ML Model?

Because we can! The model learns patterns in the melody and tries to predict pitch direction. While not useful for visualisation, it's entertaining to see the AI "anticipate" the music. Sometimes it gets the dramatic moments right, which feels magical.

## Customisation Ideas

- Adjust colours in `self.GREEN` and `self.PURPLE`
- Modify `key_change_threshold` for more/fewer sparkles
- Change ML `sequence_length` for different prediction windows
- Add more particle shapes
- Implement beat detection for rhythm-based effects

## Requirements

- Python 3.7+
- pygame 2.5.2
- mido 1.3.2
- numpy 1.24.3
- scikit-learn 1.3.2

## License

Free to use, modify, and rock out with! 🎵

---

*"Is this the real life? Is this just fantasy?"* - Now with ML!
