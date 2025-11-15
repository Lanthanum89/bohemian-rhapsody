"""
Bohemian Rhapsody Music Visualiser
A MIDI-driven visualisation with geometric bursts and ML-powered note prediction
"""

import pygame
import mido
import random
import math
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys


class NotePredictor:
    """ML model to predict if next note will be high or low (completely pointless but fun!)"""
    
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.is_trained = False
        self.median_note = 60  # Middle C as default
        
    def prepare_training_data(self, notes):
        """Prepare sequences of notes for training"""
        X, y = [], []
        
        if len(notes) < self.sequence_length + 1:
            return np.array(X), np.array(y)
        
        # Calculate median for high/low classification
        self.median_note = np.median([n['note'] for n in notes])
        
        for i in range(len(notes) - self.sequence_length):
            # Features: last N notes and their velocities
            sequence = []
            for j in range(self.sequence_length):
                sequence.extend([
                    notes[i + j]['note'],
                    notes[i + j]['velocity']
                ])
            X.append(sequence)
            
            # Label: is next note high (1) or low (0)?
            next_note = notes[i + self.sequence_length]['note']
            y.append(1 if next_note >= self.median_note else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, notes):
        """Train the model on the MIDI sequence"""
        X, y = self.prepare_training_data(notes)
        
        if len(X) < 10:  # Need minimum data
            print("Not enough notes for ML training")
            return
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        
        print(f"ML Model trained! Accuracy: {accuracy:.2%}")
        print(f"Median note value: {self.median_note}")
        self.is_trained = True
    
    def predict(self, recent_notes):
        """Predict if next note will be high or low"""
        if not self.is_trained or len(recent_notes) < self.sequence_length:
            return None
        
        # Prepare features from recent notes
        sequence = []
        for note_data in recent_notes[-self.sequence_length:]:
            sequence.extend([note_data['note'], note_data['velocity']])
        
        prediction = self.model.predict([sequence])[0]
        probability = self.model.predict_proba([sequence])[0]
        
        return {
            'prediction': 'HIGH' if prediction == 1 else 'LOW',
            'confidence': max(probability)
        }


class Particle:
    """Geometric burst particle"""
    
    def __init__(self, x, y, colour, velocity, size, shape='circle'):
        self.x = x
        self.y = y
        self.colour = colour
        self.vx, self.vy = velocity
        self.size = size
        self.shape = shape
        self.lifetime = 1.0
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-5, 5)
    
    def update(self, dt):
        """Update particle position and lifetime"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 200 * dt  # Gravity
        self.lifetime -= dt * 1.5
        self.rotation += self.rotation_speed
    
    def draw(self, screen):
        """Draw the particle"""
        if self.lifetime <= 0:
            return
        
        alpha = int(255 * self.lifetime)
        colour = (*self.colour, alpha)
        
        # Create surface with alpha
        surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        
        if self.shape == 'circle':
            pygame.draw.circle(surf, colour, (self.size, self.size), self.size)
        elif self.shape == 'square':
            # Rotate square
            points = self._get_rotated_square()
            pygame.draw.polygon(surf, colour, points)
        elif self.shape == 'triangle':
            points = self._get_rotated_triangle()
            pygame.draw.polygon(surf, colour, points)
        
        screen.blit(surf, (int(self.x - self.size), int(self.y - self.size)))
    
    def _get_rotated_square(self):
        """Get rotated square points"""
        rad = math.radians(self.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        points = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = self.size + dx * self.size * cos_r - dy * self.size * sin_r
            y = self.size + dx * self.size * sin_r + dy * self.size * cos_r
            points.append((x, y))
        return points
    
    def _get_rotated_triangle(self):
        """Get rotated triangle points"""
        rad = math.radians(self.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        points = []
        for angle in [0, 120, 240]:
            a = math.radians(angle)
            dx = math.cos(a)
            dy = math.sin(a)
            x = self.size + dx * self.size * cos_r - dy * self.size * sin_r
            y = self.size + dx * self.size * sin_r + dy * self.size * cos_r
            points.append((x, y))
        return points
    
    def is_dead(self):
        return self.lifetime <= 0


class Sparkle:
    """Special sparkle effect for key changes"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lifetime = 2.0
        self.particles = []
        
        # Create sparkle burst
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 400)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            colour = random.choice([
                (255, 255, 255),  # White
                (255, 215, 0),    # Gold
                (255, 105, 180),  # Pink
            ])
            
            size = random.uniform(2, 6)
            self.particles.append(Particle(x, y, colour, (vx, vy), size, 'circle'))
    
    def update(self, dt):
        """Update sparkle particles"""
        self.lifetime -= dt
        for particle in self.particles:
            particle.update(dt)
    
    def draw(self, screen):
        """Draw sparkle particles"""
        for particle in self.particles:
            if not particle.is_dead():
                particle.draw(screen)
    
    def is_dead(self):
        return self.lifetime <= 0


class BohemianRhapsodyVisualizer:
    """Main visualiser class"""
    
    def __init__(self, midi_file):
        pygame.init()
        # Initialize mixer with specific settings for better MIDI support
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # Screen setup
        self.width = 1200
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Bohemian Rhapsody - Music Visualiser")
        
        # Colours
        self.GREEN = (46, 204, 113)  # Emerald green
        self.PURPLE = (155, 89, 182)  # Purple
        self.BLACK = (10, 10, 15)
        
        # Load MIDI
        self.midi_file = midi_file
        self.midi = mido.MidiFile(midi_file)
        self.tempo = 500000  # Default tempo (120 BPM)
        
        # Parse all notes
        self.all_notes = self._parse_midi()
        print(f"Loaded {len(self.all_notes)} notes from MIDI")
        
        # ML Predictor
        self.predictor = NotePredictor()
        if len(self.all_notes) > 0:
            self.predictor.train(self.all_notes)
        
        # Visualisation state
        self.particles = []
        self.sparkles = []
        self.current_note_index = 0
        self.recent_notes = deque(maxlen=5)
        self.prediction = None
        
        # Timing
        self.clock = pygame.time.Clock()
        self.elapsed_time = 0
        
        # Font
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Key change detection (simplified - trigger at high intensity moments)
        self.last_key_change = 0
        self.key_change_threshold = 0.8
    
    def _parse_midi(self):
        """Parse MIDI file and extract all notes with timing"""
        notes = []
        current_time = 0
        
        for msg in mido.merge_tracks(self.midi.tracks):
            current_time += msg.time
            
            if msg.type == 'set_tempo':
                self.tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                # Convert to seconds
                time_seconds = mido.tick2second(current_time, self.midi.ticks_per_beat, self.tempo)
                notes.append({
                    'time': time_seconds,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
        
        return notes
    
    def create_burst(self, note_data):
        """Create a geometric burst based on note data"""
        velocity = note_data['velocity']
        note = note_data['note']
        
        # Position based on note pitch (low = bottom, high = top)
        y_pos = self.height - (note / 127.0) * self.height
        x_pos = random.uniform(self.width * 0.2, self.width * 0.8)
        
        # Colour based on note range
        if note < 60:  # Lower notes = purple
            colour = self.PURPLE
        else:  # Higher notes = green
            colour = self.GREEN
        
        # Number of particles based on velocity
        num_particles = int(10 + (velocity / 127.0) * 30)
        
        # Create burst
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 200) * (velocity / 127.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 100  # Initial upward velocity
            
            shape = random.choice(['circle', 'square', 'triangle'])
            size = random.uniform(3, 8) * (velocity / 127.0)
            
            self.particles.append(Particle(x_pos, y_pos, colour, (vx, vy), size, shape))
    
    def check_key_change(self, note_data):
        """Detect potential key change moments (high velocity + time gap)"""
        velocity = note_data['velocity']
        intensity = velocity / 127.0
        
        # Trigger sparkles at high intensity moments
        if intensity > self.key_change_threshold and (self.elapsed_time - self.last_key_change) > 5:
            self.last_key_change = self.elapsed_time
            
            # Create sparkles at multiple positions
            for _ in range(3):
                x = random.uniform(self.width * 0.2, self.width * 0.8)
                y = random.uniform(self.height * 0.2, self.height * 0.8)
                self.sparkles.append(Sparkle(x, y))
            
            print(f"✨ KEY CHANGE MOMENT at {self.elapsed_time:.2f}s!")
    
    def update_prediction(self):
        """Update ML prediction"""
        if len(self.recent_notes) > 0:
            self.prediction = self.predictor.predict(list(self.recent_notes))
    
    def draw_ui(self):
        """Draw UI elements"""
        # Title
        title = self.font.render("Bohemian Rhapsody", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Current note info
        if self.recent_notes:
            note_name = self._note_to_name(self.recent_notes[-1]['note'])
            velocity = self.recent_notes[-1]['velocity']
            info = self.small_font.render(f"Note: {note_name} | Velocity: {velocity}", True, (200, 200, 200))
            self.screen.blit(info, (20, 60))
        
        # ML Prediction
        if self.prediction:
            pred_colour = self.GREEN if self.prediction['prediction'] == 'HIGH' else self.PURPLE
            pred_text = f"ML Predicts: {self.prediction['prediction']} ({self.prediction['confidence']:.1%})"
            pred_surface = self.small_font.render(pred_text, True, pred_colour)
            self.screen.blit(pred_surface, (20, 90))
        
        # Progress
        if self.all_notes:
            progress = self.current_note_index / len(self.all_notes)
            pygame.draw.rect(self.screen, (50, 50, 50), (20, self.height - 40, self.width - 40, 20))
            pygame.draw.rect(self.screen, self.GREEN, (20, self.height - 40, (self.width - 40) * progress, 20))
        
        # Time
        time_text = self.small_font.render(f"Time: {self.elapsed_time:.1f}s", True, (200, 200, 200))
        self.screen.blit(time_text, (self.width - 150, 20))
    
    def _note_to_name(self, note):
        """Convert MIDI note number to name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note // 12) - 1
        note_name = notes[note % 12]
        return f"{note_name}{octave}"
    
    def run(self):
        """Main visualisation loop"""
        running = True
        start_time = pygame.time.get_ticks() / 1000.0
        
        # Try to load audio file (MP3/WAV first, then MIDI)
        audio_loaded = False
        import os
        audio_files = [
            "bohemian_rhapsody.mp3", 
            "bohemian_rhapsody.wav",
            "Queen - Bohemian Rhapsody.mp3",
            "Queen - Bohemian Rhapsody.wav"
        ]
        
        # Check for audio files
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    print(f"\n🎵 Starting visualisation with audio from {audio_file}...")
                    audio_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  {audio_file} playback failed: {e}")
        
        # Fall back to MIDI if no MP3
        if not audio_loaded:
            try:
                pygame.mixer.music.load(self.midi_file)
                pygame.mixer.music.play()
                print("\n🎵 Starting visualisation with MIDI audio...")
                print("Note: MIDI audio may not work on all systems.")
                print("For best audio, add 'bohemian_rhapsody.mp3' to this directory.\n")
                audio_loaded = True
            except Exception as e:
                print(f"\n⚠️  Audio playback failed: {e}")
                print("Continuing with visualisation only...")
                print("To add audio: Place 'bohemian_rhapsody.mp3' or 'bohemian_rhapsody.wav' in this directory.\n")
        
        print("Press ESC to exit, SPACE to pause/resume\n")
        
        paused = False
        pause_offset = 0
        pause_start = 0
        
        while running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            pygame.mixer.music.pause()
                            pause_start = pygame.time.get_ticks() / 1000.0
                        else:
                            pygame.mixer.music.unpause()
                            pause_offset += (pygame.time.get_ticks() / 1000.0) - pause_start
            
            if not paused:
                # Update elapsed time
                self.elapsed_time = (pygame.time.get_ticks() / 1000.0) - start_time - pause_offset
                
                # Process notes at current time
                while (self.current_note_index < len(self.all_notes) and 
                       self.all_notes[self.current_note_index]['time'] <= self.elapsed_time):
                    
                    note_data = self.all_notes[self.current_note_index]
                    self.create_burst(note_data)
                    self.check_key_change(note_data)
                    self.recent_notes.append(note_data)
                    self.current_note_index += 1
                    
                    # Update prediction every few notes
                    if self.current_note_index % 3 == 0:
                        self.update_prediction()
                
                # Update particles
                for particle in self.particles[:]:
                    particle.update(dt)
                    if particle.is_dead():
                        self.particles.remove(particle)
                
                # Update sparkles
                for sparkle in self.sparkles[:]:
                    sparkle.update(dt)
                    if sparkle.is_dead():
                        self.sparkles.remove(sparkle)
            
            # Draw everything
            self.screen.fill(self.BLACK)
            
            # Draw particles
            for particle in self.particles:
                particle.draw(self.screen)
            
            # Draw sparkles
            for sparkle in self.sparkles:
                sparkle.draw(self.screen)
            
            # Draw UI
            self.draw_ui()
            
            if paused:
                pause_text = self.font.render("PAUSED", True, (255, 255, 255))
                text_rect = pause_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(pause_text, text_rect)
            
            pygame.display.flip()
            
            # End when all notes are played
            if self.current_note_index >= len(self.all_notes) and len(self.particles) == 0:
                print("\n🎭 Visualisation complete!")
                pygame.mixer.music.stop()
                pygame.time.wait(2000)
                running = False
        
        pygame.mixer.music.stop()
        pygame.quit()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
    else:
        # Default file name
        midi_file = "Queen - Bohemian Rhapsody.mid"
    
    try:
        print("=" * 60)
        print("👑 BOHEMIAN RHAPSODY VISUALISER 👑")
        print("=" * 60)
        print(f"\nLoading MIDI file: {midi_file}")
        
        visualizer = BohemianRhapsodyVisualizer(midi_file)
        visualizer.run()
        
    except FileNotFoundError:
        print(f"\n❌ Error: MIDI file '{midi_file}' not found!")
        print("\nPlease provide a MIDI file:")
        print("  1. Download a MIDI version of 'Bohemian Rhapsody'")
        print("  2. Save it as 'Queen - Bohemian Rhapsody.mid' in this directory")
        print("  3. Or run: python bohemian_rhapsody_visualizer.py path/to/your/file.mid")
        print("\nYou can find MIDI files on sites like:")
        print("  - musescore.com")
        print("  - midiworld.com")
        print("  - bitmidi.com")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
