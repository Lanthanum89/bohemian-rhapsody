"""
Bohemian Rhapsody Music Visualiser
A MIDI-driven visualisation with glowing geometric bursts, a chroma wheel,
a live piano roll and an ML model that guesses whether the next note is high or low.
"""

import colorsys
import math
import os
import random
import sys
from collections import deque

import mido
import numpy as np
import pygame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DRUM_CHANNEL = 9
KICK_NOTES = {35, 36}
SNARE_NOTES = {37, 38, 39, 40}
CYMBAL_NOTES = {42, 44, 46, 49, 51, 52, 53, 55, 57, 59}

PIANO_LOW, PIANO_HIGH = 21, 108  # A0 to C8
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PURPLE = (155, 89, 182)
GREEN = (46, 204, 113)
GOLD = (255, 215, 0)
PINK = (255, 105, 180)
WHITE = (255, 255, 255)

MAX_PARTICLES = 2500


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_colour(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def scale_colour(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


def pitch_colour(note):
    """Purple for low notes, through teal, to green for high notes. Very high notes get a gold tint."""
    t = (note - 36) / (96 - 36)
    if t < 0.5:
        return lerp_colour(PURPLE, (40, 170, 200), t * 2)
    if t < 0.9:
        return lerp_colour((40, 170, 200), GREEN, (t - 0.5) / 0.4)
    return lerp_colour(GREEN, GOLD, (t - 0.9) / 0.1)


def pitch_class_colour(pc):
    r, g, b = colorsys.hsv_to_rgb(pc / 12.0, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def note_name(note):
    return f"{NOTE_NAMES[note % 12]}{(note // 12) - 1}"


def fmt_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class GlowCache:
    """Pre-rendered radial glow sprites, blitted additively. Avoids allocating a surface per particle per frame."""

    def __init__(self):
        self._cache = {}

    def get(self, colour, radius, brightness=1.0):
        radius = max(2, int(radius))
        level = max(1, min(8, int(brightness * 8 + 0.5)))
        colour = tuple((c // 16) * 16 for c in colour)
        key = (colour, radius, level)
        sprite = self._cache.get(key)
        if sprite is None:
            size = radius * 2
            y, x = np.ogrid[-radius:radius, -radius:radius]
            dist = np.sqrt(x * x + y * y) / radius
            falloff = np.clip(1.0 - dist, 0, 1) ** 2 * (level / 8.0)
            rgb = np.zeros((size, size, 3), dtype=np.uint8)
            for i in range(3):
                rgb[..., i] = (falloff * colour[i]).astype(np.uint8)
            sprite = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[key] = sprite
        return sprite


GLOW = GlowCache()


class NotePredictor:
    """ML model to predict if next note will be high or low (completely pointless but fun!)"""

    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        self.is_trained = False
        self.median_note = 60
        self.test_accuracy = None

    def prepare_training_data(self, notes):
        X, y = [], []
        if len(notes) < self.sequence_length + 1:
            return np.array(X), np.array(y)

        self.median_note = float(np.median([n['note'] for n in notes]))
        for i in range(len(notes) - self.sequence_length):
            sequence = []
            for j in range(self.sequence_length):
                sequence.extend([notes[i + j]['note'], notes[i + j]['velocity']])
            X.append(sequence)
            y.append(1 if notes[i + self.sequence_length]['note'] >= self.median_note else 0)
        return np.array(X), np.array(y)

    def train(self, notes):
        X, y = self.prepare_training_data(notes)
        if len(X) < 10:
            print("Not enough notes for ML training")
            return

        # shuffle=False: consecutive windows overlap, so a shuffled split leaks the test set into training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.model.fit(X_train, y_train)
        self.test_accuracy = self.model.score(X_test, y_test)
        print(f"ML Model trained! Held-out accuracy (last 20% of song): {self.test_accuracy:.2%}")
        print(f"Median note value: {self.median_note}")
        self.is_trained = True

    def predict(self, recent_notes):
        if not self.is_trained or len(recent_notes) < self.sequence_length:
            return None
        sequence = []
        for n in list(recent_notes)[-self.sequence_length:]:
            sequence.extend([n['note'], n['velocity']])
        probability = self.model.predict_proba([sequence])[0]
        is_high = probability[-1] >= 0.5 if len(probability) > 1 else bool(self.model.classes_[0])
        return {'prediction': 'HIGH' if is_high else 'LOW', 'confidence': float(max(probability))}


class Particle:
    """Glowing geometric burst particle"""

    def __init__(self, x, y, colour, velocity, size, shape='circle', decay=1.5, gravity=200, glow=True):
        self.x, self.y = x, y
        self.colour = colour
        self.vx, self.vy = velocity
        self.size = size
        self.shape = shape
        self.lifetime = 1.0
        self.decay = decay
        self.gravity = gravity
        self.glow = glow
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-300, 300)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        self.vx *= 0.99
        self.lifetime -= dt * self.decay
        self.rotation += self.rotation_speed * dt

    def draw(self, trail, glow_layer):
        if self.lifetime <= 0:
            return
        life = self.lifetime
        colour = scale_colour(self.colour, 0.4 + 0.6 * life)
        size = max(1.0, self.size * (0.5 + 0.5 * life))

        if self.glow:
            g = GLOW.get(self.colour, size * 3.5, life * 0.8)
            glow_layer.blit(g, (self.x - g.get_width() / 2, self.y - g.get_height() / 2),
                            special_flags=pygame.BLEND_ADD)

        if self.shape == 'circle':
            pygame.draw.circle(trail, colour, (int(self.x), int(self.y)), int(size))
        else:
            corners = 4 if self.shape == 'square' else 3
            rad = math.radians(self.rotation)
            pts = [(self.x + math.cos(rad + k * 2 * math.pi / corners) * size,
                    self.y + math.sin(rad + k * 2 * math.pi / corners) * size) for k in range(corners)]
            pygame.draw.polygon(trail, colour, pts)

    def is_dead(self):
        return self.lifetime <= 0


class Sparkle:
    """Dramatic-moment sparkle explosion"""

    def __init__(self, x, y):
        self.lifetime = 2.0
        self.particles = []
        for _ in range(80):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(100, 500)
            colour = random.choice([WHITE, GOLD, PINK])
            self.particles.append(Particle(x, y, colour, (math.cos(angle) * speed, math.sin(angle) * speed),
                                           random.uniform(2, 5), random.choice(['circle', 'circle', 'triangle']),
                                           decay=0.6, gravity=80))

    def update(self, dt):
        self.lifetime -= dt
        for p in self.particles:
            p.update(dt)

    def draw(self, trail, glow_layer):
        for p in self.particles:
            if not p.is_dead():
                p.draw(trail, glow_layer)

    def is_dead(self):
        return self.lifetime <= 0


class Shockwave:
    """Expanding ring for loud notes and snare hits"""

    def __init__(self, x, y, colour, max_radius=220, width=3, speed=1.6):
        self.x, self.y = x, y
        self.colour = colour
        self.max_radius = max_radius
        self.width = width
        self.speed = speed
        self.t = 0.0

    def update(self, dt):
        self.t += dt * self.speed

    def draw(self, surface):
        if self.t >= 1:
            return
        ease = 1 - (1 - self.t) ** 3
        r = int(10 + ease * self.max_radius)
        pygame.draw.circle(surface, scale_colour(self.colour, 1 - self.t), (int(self.x), int(self.y)), r,
                           max(1, int(self.width * (1 - self.t) + 1)))

    def is_dead(self):
        return self.t >= 1


class Star:
    def __init__(self, w, h):
        self.x = random.uniform(0, w)
        self.y = random.uniform(0, h)
        self.depth = random.uniform(0.2, 1.0)
        self.phase = random.uniform(0, 2 * math.pi)


class BohemianRhapsodyVisualizer:
    """Main visualiser class"""

    def __init__(self, midi_file, headless=False):
        pygame.init()
        if not headless:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            except pygame.error as e:
                print(f"⚠️  Audio device unavailable: {e}")

        self.width, self.height = 1280, 800
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.SCALED)
        pygame.display.set_caption("Bohemian Rhapsody - Music Visualiser")

        self.midi_file = midi_file
        self.midi = mido.MidiFile(midi_file)
        self.all_notes, self.tempo_changes = self._parse_midi()
        self.melodic_notes = [n for n in self.all_notes if n['channel'] != DRUM_CHANNEL]
        self.duration = max((n['end'] for n in self.all_notes), default=0)
        print(f"Loaded {len(self.all_notes)} notes ({len(self.melodic_notes)} melodic) "
              f"over {fmt_time(self.duration)}")

        # ML on melodic notes only: drum note numbers are instruments, not pitches
        self.predictor = NotePredictor()
        if self.melodic_notes:
            self.predictor.train(self.melodic_notes)

        # Layers
        self.trail = pygame.Surface((self.width, self.height))
        self.glow_layer = pygame.Surface((self.width, self.height))
        self.hud = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.gradients = [self._make_gradient(i / 15) for i in range(16)]
        # Blitting a pre-filled surface with BLEND_MULT/ADD hits SDL's SIMD path; Surface.fill with
        # the same flags is ~25x slower and blew the 60 FPS budget on its own.
        self.trail_fade = pygame.Surface((self.width, self.height))
        self.trail_fade.fill((218, 218, 222))
        # Multiply alone rounds dim pixels back up and leaves a permanent haze; subtract finishes them off
        self.trail_floor = pygame.Surface((self.width, self.height))
        self.trail_floor.fill((3, 3, 3))
        self.flash_surface = pygame.Surface((self.width, self.height))
        self.stars = [Star(self.width, self.height) for _ in range(180)]

        # Visual state
        self.particles = []
        self.sparkles = []
        self.shockwaves = []
        self.chroma = [0.0] * 12
        self.activity = 0.0        # velocity-weighted notes per ~second
        self.energy = 0.0          # activity normalised to 0..1, drives background mood
        self.bass_pulse = 0.0      # kick drum
        self.flash = 0.0           # sparkle-moment screen flash
        self.shake = 0.0
        self.wheel_rotation = 0.0
        self.show_hud = True
        self.show_trails = True

        # Playback state
        self.current_note_index = 0
        self.recent_notes = deque(maxlen=5)
        self.prediction = None
        self.pending_prediction = None
        self.prediction_hits = 0
        self.prediction_total = 0
        self.prediction_history = deque(maxlen=40)
        self.elapsed_time = 0.0
        self.last_key_change = -10
        self.key_change_threshold = 0.8
        self.dramatic_energy = 0.75

        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 44)
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

    # ------------------------------------------------------------------ parsing

    def _parse_midi(self):
        """Parse notes with correct timing across tempo changes.

        Iterating a MidiFile yields messages with .time already in seconds, using every tempo
        change in order. (Converting cumulative ticks with the *latest* tempo, as the old version
        did, shifts every note once the song changes tempo.)
        """
        notes, tempo_changes, open_notes = [], [], {}
        t = 0.0
        for msg in self.midi:
            t += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append((t, mido.tempo2bpm(msg.tempo)))
            elif msg.type == 'note_on' and msg.velocity > 0:
                n = {'time': t, 'end': t + 0.3, 'note': msg.note, 'velocity': msg.velocity, 'channel': msg.channel}
                notes.append(n)
                open_notes.setdefault((msg.channel, msg.note), []).append(n)
            elif msg.type in ('note_off', 'note_on'):
                stack = open_notes.get((msg.channel, msg.note))
                if stack:
                    stack.pop(0)['end'] = t
        return notes, tempo_changes

    # ----------------------------------------------------------------- building

    def _make_gradient(self, mood):
        top = lerp_colour((8, 6, 20), (40, 6, 40), mood)
        bottom = lerp_colour((14, 10, 34), (70, 12, 50), mood)
        col = pygame.Surface((1, self.height))
        for y in range(self.height):
            col.set_at((0, y), lerp_colour(top, bottom, y / self.height))
        return pygame.transform.scale(col, (self.width, self.height))

    def create_burst(self, note_data):
        velocity = note_data['velocity'] / 127.0
        note = note_data['note']
        y_pos = self.height * 0.88 - ((note - 24) / 84.0) * self.height * 0.78
        x_pos = random.uniform(self.width * 0.12, self.width * 0.88)
        colour = pitch_colour(note)

        for _ in range(int(8 + velocity * 26)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(60, 240) * velocity
            self.particles.append(Particle(
                x_pos, y_pos, colour,
                (math.cos(angle) * speed, math.sin(angle) * speed - 90),
                random.uniform(2.5, 7) * (0.5 + velocity),
                random.choice(['circle', 'square', 'triangle']),
                glow=random.random() < 0.35))

        if velocity > 0.88 and len(self.shockwaves) < 6:
            self.shockwaves.append(Shockwave(x_pos, y_pos, colour, max_radius=120 + 160 * velocity))

    def handle_drum(self, note_data):
        note, velocity = note_data['note'], note_data['velocity'] / 127.0
        if note in KICK_NOTES:
            self.bass_pulse = min(1.0, self.bass_pulse + 0.6 * velocity + 0.2)
        elif note in SNARE_NOTES:
            self.shockwaves.append(Shockwave(self.width / 2, self.height / 2, (180, 160, 255),
                                             max_radius=420, width=2, speed=2.2))
        elif note in CYMBAL_NOTES:
            for _ in range(6):
                self.particles.append(Particle(
                    random.uniform(0, self.width), random.uniform(0, self.height * 0.5), (200, 220, 255),
                    (random.uniform(-20, 20), random.uniform(-20, 20)), random.uniform(1, 2.2),
                    'circle', decay=2.5, gravity=0))

    def check_key_change(self, note_data):
        """Dramatic moments: a loud melodic note while overall energy is high, at most every 15s"""
        intensity = note_data['velocity'] / 127.0
        if (intensity > self.key_change_threshold and self.energy > self.dramatic_energy
                and (self.elapsed_time - self.last_key_change) > 15):
            self.last_key_change = self.elapsed_time
            for _ in range(3):
                self.sparkles.append(Sparkle(random.uniform(self.width * 0.2, self.width * 0.8),
                                             random.uniform(self.height * 0.2, self.height * 0.7)))
            self.flash = 0.6
            self.shake = 12
            print(f"✨ DRAMATIC MOMENT at {fmt_time(self.elapsed_time)}")

    def _score_prediction(self, note_data):
        if self.pending_prediction is None:
            return
        actual = 'HIGH' if note_data['note'] >= self.predictor.median_note else 'LOW'
        hit = actual == self.pending_prediction['prediction']
        self.prediction_hits += hit
        self.prediction_total += 1
        self.prediction_history.append(hit)
        self.pending_prediction = None

    # ------------------------------------------------------------------- update

    def update(self, dt):
        new_melodic = False
        while (self.current_note_index < len(self.all_notes) and
               self.all_notes[self.current_note_index]['time'] <= self.elapsed_time):
            n = self.all_notes[self.current_note_index]
            self.current_note_index += 1
            self.activity += n['velocity'] / 127.0
            if n['channel'] == DRUM_CHANNEL:
                self.handle_drum(n)
                continue
            self._score_prediction(n)
            self.create_burst(n)
            self.check_key_change(n)
            self.recent_notes.append(n)
            self.chroma[n['note'] % 12] = min(1.5, self.chroma[n['note'] % 12] + 0.25 + n['velocity'] / 254)
            new_melodic = True

        if new_melodic:
            self.prediction = self.predictor.predict(self.recent_notes)
            self.pending_prediction = self.prediction

        if len(self.particles) > MAX_PARTICLES:
            del self.particles[:len(self.particles) - MAX_PARTICLES]

        for group in (self.particles, self.sparkles, self.shockwaves):
            for item in group:
                item.update(dt)
            group[:] = [i for i in group if not i.is_dead()]

        decay = math.exp(-dt * 2.5)
        self.chroma = [c * decay for c in self.chroma]
        self.activity *= math.exp(-dt * 1.5)
        target = min(1.0, self.activity / 30.0)  # ~30 = the busiest 5% of this arrangement
        self.energy += (target - self.energy) * min(1.0, dt * 3)
        self.bass_pulse *= math.exp(-dt * 7)
        self.flash *= math.exp(-dt * 4)
        self.shake *= math.exp(-dt * 8)
        self.wheel_rotation += dt * (8 + self.energy * 40)

    def is_finished(self):
        return (self.current_note_index >= len(self.all_notes)
                and not self.particles and not self.sparkles)

    # --------------------------------------------------------------------- draw

    def draw(self, paused=False):
        t = self.elapsed_time
        ox = random.uniform(-self.shake, self.shake)
        oy = random.uniform(-self.shake, self.shake)

        # Background mood gradient + bass pulse
        self.screen.blit(self.gradients[int(self.energy * 15)], (0, 0))
        if self.bass_pulse > 0.02:
            pulse = GLOW.get((120, 40, 160), 300 + 200 * self.bass_pulse, self.bass_pulse * 0.7)
            self.screen.blit(pulse, (self.width / 2 - pulse.get_width() / 2 + ox,
                                     self.height / 2 - pulse.get_height() / 2 + oy),
                             special_flags=pygame.BLEND_ADD)

        # Starfield (parallax drift, twinkle, nudged by the kick)
        for s in self.stars:
            x = (s.x - t * 12 * s.depth) % self.width
            twinkle = 0.55 + 0.45 * math.sin(t * 2.5 * s.depth + s.phase)
            b = int(min(255, (60 + 150 * s.depth) * twinkle + 80 * self.bass_pulse))
            self.screen.set_at((int(x), int(s.y)), (b, b, min(255, b + 30)))
            if s.depth > 0.85:
                pygame.draw.circle(self.screen, (b // 2, b // 2, b // 2 + 20), (int(x), int(s.y)), 2)

        # Trails: fade the persistent layer instead of clearing it
        if self.show_trails:
            self.trail.blit(self.trail_fade, (0, 0), special_flags=pygame.BLEND_MULT)
            self.trail.blit(self.trail_floor, (0, 0), special_flags=pygame.BLEND_SUB)
        else:
            self.trail.fill((0, 0, 0))
        self.glow_layer.fill((0, 0, 0))

        self._draw_chroma_wheel(self.glow_layer)
        for w in self.shockwaves:
            w.draw(self.glow_layer)
        for p in self.particles:
            p.draw(self.trail, self.glow_layer)
        for s in self.sparkles:
            s.draw(self.trail, self.glow_layer)

        self.screen.blit(self.trail, (ox, oy), special_flags=pygame.BLEND_ADD)
        self.screen.blit(self.glow_layer, (ox, oy), special_flags=pygame.BLEND_ADD)

        if self.flash > 0.02:
            v = int(255 * self.flash * 0.5)
            self.flash_surface.fill((v, v, int(v * 0.8)))
            self.screen.blit(self.flash_surface, (0, 0), special_flags=pygame.BLEND_ADD)

        self._draw_piano()
        if self.show_hud:
            self._draw_hud(paused)
        elif paused:
            self._draw_paused()

    def _draw_chroma_wheel(self, surface):
        """12 pitch classes around a ring. Bars grow with how much each pitch is being played."""
        cx, cy = self.width / 2, self.height * 0.45
        base_r = 70 + 25 * self.bass_pulse
        for pc in range(12):
            energy = min(1.0, self.chroma[pc])
            angle = math.radians(self.wheel_rotation + pc * 30 - 90)
            colour = pitch_class_colour(pc)
            length = 8 + energy * 110
            for side in (-1, 1):  # two thin wedges per pitch class read as a petal
                a1 = angle + side * math.radians(9)
                pts = [(cx + math.cos(angle) * base_r, cy + math.sin(angle) * base_r),
                       (cx + math.cos(a1) * (base_r + length * 0.6), cy + math.sin(a1) * (base_r + length * 0.6)),
                       (cx + math.cos(angle) * (base_r + length), cy + math.sin(angle) * (base_r + length))]
                pygame.draw.polygon(surface, scale_colour(colour, 0.25 + 0.75 * energy), pts)
            if energy > 0.15:
                g = GLOW.get(colour, 20 + energy * 30, energy * 0.6)
                ex = cx + math.cos(angle) * (base_r + length)
                ey = cy + math.sin(angle) * (base_r + length)
                self.glow_layer.blit(g, (ex - g.get_width() / 2, ey - g.get_height() / 2),
                                     special_flags=pygame.BLEND_ADD)
        pygame.draw.circle(surface, scale_colour((150, 120, 220), 0.4 + 0.6 * self.bass_pulse),
                           (int(cx), int(cy)), int(base_r - 6), 2)

    def _draw_piano(self):
        """88-key strip along the bottom, lit by every note currently sounding"""
        top, h = self.height - 70, 46
        white_keys = [n for n in range(PIANO_LOW, PIANO_HIGH + 1) if n % 12 not in (1, 3, 6, 8, 10)]
        kw = (self.width - 40) / len(white_keys)
        active = {}
        t = self.elapsed_time
        # Notes are sorted by start time; scan back from the playhead for ones still sounding
        i = self.current_note_index - 1
        while i >= 0 and t - self.all_notes[i]['time'] < 12:
            n = self.all_notes[i]
            if n['channel'] != DRUM_CHANNEL and n['end'] > t:
                active.setdefault(n['note'], n)
            i -= 1

        positions = {}
        for idx, n in enumerate(white_keys):
            x = 20 + idx * kw
            positions[n] = x
            colour = (40, 40, 52)
            if n in active:
                colour = pitch_colour(n)
                g = GLOW.get(colour, 26, 0.9)
                self.screen.blit(g, (x + kw / 2 - 26, top - 30), special_flags=pygame.BLEND_ADD)
            pygame.draw.rect(self.screen, colour, (x + 1, top, kw - 2, h), border_radius=3)
        for n in range(PIANO_LOW, PIANO_HIGH + 1):
            if n % 12 in (1, 3, 6, 8, 10):
                x = positions[n - 1] + kw * 0.65
                colour = pitch_colour(n) if n in active else (14, 14, 20)
                pygame.draw.rect(self.screen, colour, (x, top, kw * 0.7, h * 0.6), border_radius=2)

    def _panel(self, rect):
        pygame.draw.rect(self.hud, (10, 8, 24, 170), rect, border_radius=12)
        pygame.draw.rect(self.hud, (120, 90, 200, 90), rect, 1, border_radius=12)

    def _draw_hud(self, paused):
        self.hud.fill((0, 0, 0, 0))
        hud = self.hud

        # Top-left: title and current note
        self._panel((16, 16, 330, 92))
        hud.blit(self.font_big.render("Bohemian Rhapsody", True, WHITE), (30, 28))
        if self.recent_notes:
            n = self.recent_notes[-1]
            hud.blit(self.font.render(note_name(n['note']), True, pitch_colour(n['note'])), (30, 70))
            vel_w = int(200 * n['velocity'] / 127)
            pygame.draw.rect(hud, (60, 60, 80), (110, 76, 200, 8), border_radius=4)
            pygame.draw.rect(hud, pitch_colour(n['note']), (110, 76, vel_w, 8), border_radius=4)

        # Top-right: ML panel with confidence gauge and live hit history
        px, pw = self.width - 346, 330
        self._panel((px, 16, pw, 128))
        hud.blit(self.font_small.render("ML NEXT-NOTE GUESS", True, (170, 160, 210)), (px + 14, 26))
        if self.prediction:
            high = self.prediction['prediction'] == 'HIGH'
            col = GREEN if high else PURPLE
            ax, ay = px + 34, 72
            arrow = [(ax, ay - 16), (ax - 14, ay + 4), (ax + 14, ay + 4)] if high else \
                    [(ax, ay + 16), (ax - 14, ay - 4), (ax + 14, ay - 4)]
            pygame.draw.polygon(hud, col, arrow)
            hud.blit(self.font.render(f"{self.prediction['prediction']}  {self.prediction['confidence']:.0%}",
                                      True, col), (px + 62, 60))
            conf_w = int((pw - 90) * self.prediction['confidence'])
            pygame.draw.rect(hud, (60, 60, 80), (px + 62, 86, pw - 90, 6), border_radius=3)
            pygame.draw.rect(hud, col, (px + 62, 86, conf_w, 6), border_radius=3)
        for i, hit in enumerate(self.prediction_history):
            pygame.draw.rect(hud, GREEN if hit else (200, 60, 90), (px + 14 + i * 7.6, 104, 5, 12), border_radius=2)
        if self.prediction_total:
            rate = self.prediction_hits / self.prediction_total
            hud.blit(self.font_small.render(f"live hit rate {rate:.0%} of {self.prediction_total}",
                                            True, (190, 190, 210)), (px + 14, 122))

        # Progress bar with tempo-change markers
        bx, by, bw = 20, self.height - 16, self.width - 40
        progress = min(1.0, self.elapsed_time / self.duration) if self.duration else 0
        pygame.draw.rect(hud, (50, 50, 64, 200), (bx, by, bw, 6), border_radius=3)
        bar_col = lerp_colour(PURPLE, GREEN, progress)
        pygame.draw.rect(hud, bar_col, (bx, by, int(bw * progress), 6), border_radius=3)
        for ct, _bpm in self.tempo_changes[1:]:
            mx = bx + bw * ct / self.duration
            pygame.draw.line(hud, GOLD, (mx, by - 4), (mx, by + 9), 2)
        time_txt = f"{fmt_time(self.elapsed_time)} / {fmt_time(self.duration)}"
        bpm = next((b for ct, b in reversed(self.tempo_changes) if ct <= self.elapsed_time), None)
        if bpm:
            time_txt += f"   {bpm:.0f} BPM"
        hud.blit(self.font_small.render(time_txt, True, (190, 190, 210)), (22, self.height - 90))
        keys = self.font_small.render("SPACE pause   F fullscreen   H hud   T trails   ESC quit",
                                      True, (120, 120, 150))
        hud.blit(keys, (self.width - keys.get_width() - 22, self.height - 90))

        if paused:
            self._draw_paused(hud)
        self.screen.blit(hud, (0, 0))

    def _draw_paused(self, surface=None):
        surface = surface or self.screen
        txt = self.font_big.render("PAUSED", True, WHITE)
        surface.blit(txt, txt.get_rect(center=(self.width // 2, self.height * 0.45)))

    # ---------------------------------------------------------------------- run

    def _start_audio(self):
        if not pygame.mixer.get_init():
            print("Continuing with visualisation only...")
            return
        midi_dir = os.path.dirname(os.path.abspath(self.midi_file))
        candidates = ["bohemian_rhapsody.mp3", "bohemian_rhapsody.wav",
                      "Queen - Bohemian Rhapsody.mp3", "Queen - Bohemian Rhapsody.wav"]
        for name in candidates:
            for path in (name, os.path.join(midi_dir, name)):
                if os.path.exists(path):
                    try:
                        pygame.mixer.music.load(path)
                        pygame.mixer.music.play()
                        print(f"\n🎵 Playing audio from {path}")
                        return
                    except pygame.error as e:
                        print(f"⚠️  {path} playback failed: {e}")
        try:
            pygame.mixer.music.load(self.midi_file)
            pygame.mixer.music.play()
            print("\n🎵 Playing MIDI audio (may be silent on systems without a MIDI synth).")
            print("For best audio, add 'bohemian_rhapsody.mp3' to this directory.\n")
        except pygame.error as e:
            print(f"\n⚠️  Audio playback failed: {e}")
            print("Continuing with visualisation only...\n")

    def run(self):
        self._start_audio()
        print("SPACE pause | F fullscreen | H toggle HUD | T toggle trails | ESC exit\n")

        start = pygame.time.get_ticks() / 1000.0
        paused, pause_offset, pause_start = False, 0.0, 0.0
        running = True

        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        now = pygame.time.get_ticks() / 1000.0
                        if paused:
                            pause_start = now
                            if pygame.mixer.get_init():
                                pygame.mixer.music.pause()
                        else:
                            pause_offset += now - pause_start
                            if pygame.mixer.get_init():
                                pygame.mixer.music.unpause()
                    elif event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_h:
                        self.show_hud = not self.show_hud
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails

            if not paused:
                self.elapsed_time = pygame.time.get_ticks() / 1000.0 - start - pause_offset
                self.update(dt)

            self.draw(paused)
            pygame.display.flip()

            if self.is_finished():
                print("\n🎭 Visualisation complete!")
                pygame.time.wait(1500)
                running = False

        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        pygame.quit()


def main():
    midi_file = sys.argv[1] if len(sys.argv) > 1 else "Queen - Bohemian Rhapsody.mid"
    try:
        print("=" * 60)
        print("👑 BOHEMIAN RHAPSODY VISUALISER 👑")
        print("=" * 60)
        print(f"\nLoading MIDI file: {midi_file}")
        BohemianRhapsodyVisualizer(midi_file).run()
    except FileNotFoundError:
        print(f"\n❌ Error: MIDI file '{midi_file}' not found!")
        print("\nPlease provide a MIDI file:")
        print("  1. Download a MIDI version of 'Bohemian Rhapsody'")
        print("  2. Save it as 'Queen - Bohemian Rhapsody.mid' in this directory")
        print("  3. Or run: python bohemian_rhapsody_visualizer.py path/to/your/file.mid")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
