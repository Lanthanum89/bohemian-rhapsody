# Audio Setup Instructions

To hear the MIDI audio properly, you need to install FluidSynth:

## Option 1: Install FluidSynth (Recommended for best audio quality)

### Windows:
1. Download FluidSynth from: https://github.com/FluidSynth/fluidsynth/releases
2. Extract and add to your PATH, or install via Chocolatey:
   ```bash
   choco install fluidsynth
   ```

3. Download a SoundFont file (e.g., GeneralUser GS):
   - https://schristiancollins.com/generaluser.php
   - Save as `soundfont.sf2` in this directory

## Option 2: Use pygame's built-in MIDI (Current setup)

The current setup uses pygame's MIDI playback which should work on most systems but may:
- Sound lower quality than FluidSynth
- Not work on all Windows systems without additional MIDI drivers
- Require Windows MIDI Mapper configuration

## Option 3: Use an MP3 file

For the best experience, download or convert an MP3 version of Bohemian Rhapsody:
1. Save as `bohemian_rhapsody.mp3` in this directory
2. The visualizer will automatically use it if available

## Testing

Run the visualizer:
```bash
python bohemian_rhapsody_visualizer.py
```

If you don't hear audio, the visuals will still work perfectly!
