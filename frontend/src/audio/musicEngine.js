// Generative calming music with Tone.js.
// The backend decides *what* to play (preset, tempo, scale, loudness, density);
// this turns it into sound and changes it smoothly so transitions never jar.
import * as Tone from 'tone';

export const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  pentatonic_major: [0, 2, 4, 7, 9],
  pentatonic_minor: [0, 3, 5, 7, 10],
};

const ROOT_MIDI = 57; // A3
const PROGRESSIONS = { 7: [0, 5, 3, 4], 5: [0, 3, 1, 4] }; // scale-degree roots per bar

function buildInstrument(preset) {
  switch (preset) {
    case 'ambient_pad':
      return new Tone.PolySynth(Tone.AMSynth, {
        harmonicity: 1.5,
        envelope: { attack: 1.8, decay: 0.5, sustain: 0.8, release: 4 },
        modulationEnvelope: { attack: 2, release: 4 },
      });
    case 'soft_strings':
      return new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'fatsawtooth', count: 3, spread: 18 },
        envelope: { attack: 0.9, decay: 0.3, sustain: 0.7, release: 2.5 },
      });
    case 'pentatonic_flute':
      return new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'sine' },
        envelope: { attack: 0.12, decay: 0.2, sustain: 0.6, release: 1.2 },
      });
    case 'mellow_keys':
      return new Tone.PolySynth(Tone.FMSynth, {
        harmonicity: 3.01,
        modulationIndex: 2,
        envelope: { attack: 0.005, decay: 1.6, sustain: 0.05, release: 1.6 },
      });
    case 'warm_piano':
    default:
      return new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.01, decay: 1.4, sustain: 0.15, release: 1.8 },
      });
  }
}

const midiToFreq = (m) => Tone.Frequency(m, 'midi').toFrequency();

class MusicEngine {
  constructor() {
    this.started = false;
    this.params = null;
    this.instrument = null;
    this.preset = null;
    this.bar = 0;
    this.beat = 0;
    this.melodyIdx = 7;
  }

  async init() {
    if (this.started) return;
    await Tone.start(); // must be called from a user gesture (the Start button)
    this.master = new Tone.Volume(-60).toDestination();
    this.reverb = new Tone.Reverb({ decay: 5, wet: 0.35 }).connect(this.master);
    this.filter = new Tone.Filter(2200, 'lowpass').connect(this.reverb);
    this.bass = new Tone.Synth({
      oscillator: { type: 'sine' },
      envelope: { attack: 0.4, decay: 0.5, sustain: 0.6, release: 2 },
      volume: -14,
    }).connect(this.reverb);
    this.loop = new Tone.Loop((time) => this._tick(time), '4n');
    this.started = true;
  }

  _scale() {
    return SCALES[this.params?.scale] || SCALES.pentatonic_major;
  }

  _note(degree, octave = 0) {
    const s = this._scale();
    const oct = Math.floor(degree / s.length) + octave;
    const idx = ((degree % s.length) + s.length) % s.length;
    return ROOT_MIDI + s[idx] + 12 * oct;
  }

  _tick(time) {
    if (!this.params || !this.instrument) return;
    const s = this._scale();
    const prog = PROGRESSIONS[s.length] || PROGRESSIONS[7];
    const chordRoot = prog[this.bar % prog.length];
    const vel = Math.max(0.1, Math.min(1, this.params.velocity ?? 0.5));

    if (this.beat === 0) {
      // chord (root, third, fifth in scale steps) + bass, once per bar
      const chord = [0, 2, 4].map((d) => midiToFreq(this._note(chordRoot + d)));
      this.instrument.triggerAttackRelease(chord, '1m', time, vel * 0.45);
      this.bass.triggerAttackRelease(midiToFreq(this._note(chordRoot, -1)), '1m', time, vel * 0.6);
    }
    // sparse melody: a gentle random walk over the scale
    if (Math.random() < (this.params.density ?? 0.4)) {
      const step = [-2, -1, -1, 0, 1, 1, 2][Math.floor(Math.random() * 7)];
      this.melodyIdx = Math.max(s.length, Math.min(s.length * 3 - 1, this.melodyIdx + step));
      if (Math.random() < 0.3) this.melodyIdx = chordRoot + s.length * 2; // resolve to chord tone
      this.instrument.triggerAttackRelease(midiToFreq(this._note(this.melodyIdx)), '4n', time, vel * 0.55);
    }
    this.beat = (this.beat + 1) % 4;
    if (this.beat === 0) this.bar += 1;
  }

  _swapInstrument(preset) {
    const old = this.instrument;
    const next = buildInstrument(preset);
    next.volume.value = -40;
    next.connect(this.filter);
    next.volume.rampTo(-8, 4);
    if (old) {
      old.volume.rampTo(-60, 4);
      setTimeout(() => old.dispose(), 5000);
    }
    this.instrument = next;
    this.preset = preset;
  }

  // params: {tempo_bpm, scale, velocity, density, preset} or null for silence
  setParams(params) {
    if (!this.started) return;
    if (!params || !params.preset) {
      this.params = null;
      this.master.volume.rampTo(-60, 3);
      return;
    }
    const firstStart = !this.params;
    this.params = params;
    if (params.preset !== this.preset) this._swapInstrument(params.preset);
    const tempo = Math.max(40, Math.min(140, params.tempo_bpm || 70));
    if (firstStart) Tone.Transport.bpm.value = tempo;
    else Tone.Transport.bpm.rampTo(tempo, 8);
    this.master.volume.rampTo(-26 + 14 * (params.velocity ?? 0.5), firstStart ? 4 : 6);
    if (Tone.Transport.state !== 'started') {
      this.loop.start(0);
      Tone.Transport.start();
    }
  }

  stop() {
    if (!this.started) return;
    this.params = null;
    this.master.volume.rampTo(-60, 2);
    setTimeout(() => {
      Tone.Transport.stop();
      this.loop.stop();
    }, 2200);
  }
}

export const musicEngine = new MusicEngine();
