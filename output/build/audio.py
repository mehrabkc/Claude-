"""Synthesises the funky 120 BPM backing track + all SFX. Writes music.wav and sfx.wav (stereo, 44.1k)."""
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
from timeline import *

SR = 44100
N = int(SR * DUR)
rng = np.random.default_rng(7)
S16 = BEAT / 4  # one sixteenth


def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)
def env_exp(n, tau): return np.exp(-np.arange(n) / (tau * SR))
def tarr(d): return np.arange(int(d * SR)) / SR


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f, order=2): return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)
def lp(x, f, order=2): return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x)


def svf_lp(x, cutoff, q=0.9):
    """state-variable low-pass with per-sample cutoff (array). q ~ 1/resonance."""
    out = np.empty_like(x)
    low = band = 0.0
    f = 2 * np.sin(np.pi * np.clip(cutoff, 20, SR * 0.22) / SR)
    for i in range(len(x)):
        low += f[i] * band
        high = x[i] - low - q * band
        band += f[i] * high
        out[i] = low
    return out


def saw(freq, n, detune=0.0):
    ph = np.cumsum(np.full(n, freq * (1 + detune) / SR)) % 1.0
    return 2 * ph - 1
def sqr(freq, n, pw=0.5):
    ph = np.cumsum(np.full(n, freq / SR)) % 1.0
    return np.where(ph < pw, 1.0, -1.0)


def place(buf, sig, t, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= len(buf): return
    sig = sig[: len(buf) - i] if i >= 0 else sig[-i:]
    i = max(i, 0)
    l = gain * np.cos((pan + 1) * np.pi / 4); r = gain * np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * l
    buf[i:i + len(sig), 1] += sig * r


# ------------------------------------------------------------------ drums
def kick():
    n = int(0.38 * SR); t = np.arange(n) / SR
    f = 45 + 120 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.16)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.25
    return np.tanh(1.6 * (body + click))
KICK = kick()

def clap():
    n = int(0.28 * SR); out = np.zeros(n)
    for off in (0, 0.011, 0.022):
        i = int(off * SR); m = n - i
        out[i:] += bp(rng.standard_normal(m), 900, 3800) * np.exp(-np.arange(m) / (0.012 * SR))
    tail = bp(rng.standard_normal(n), 1000, 3500) * np.exp(-np.arange(n) / (0.07 * SR))
    return np.tanh(1.5 * (out * 0.6 + tail * 0.6))
CLAP = clap()

def hat(d, tau):
    n = int(d * SR)
    return hp(rng.standard_normal(n), 7000, 3) * np.exp(-np.arange(n) / (tau * SR))
HAT_C = hat(0.08, 0.016); HAT_O = hat(0.3, 0.08); SHK = hat(0.06, 0.012)

def snare():
    n = int(0.25 * SR); t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * (190 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.05)
    nz = bp(rng.standard_normal(n), 1500, 7000) * np.exp(-t / 0.07)
    return np.tanh(1.3 * (tone * 0.7 + nz))
SNARE = snare()

def crash(d=1.6):
    n = int(d * SR)
    return hp(rng.standard_normal(n), 4500) * np.exp(-np.arange(n) / (0.45 * SR))


# ------------------------------------------------------------------ pitched
def bass_note(m, d, vel=1.0):
    n = int(d * SR); f = mtof(m)
    x = 0.6 * saw(f, n) + 0.5 * sqr(f, n, 0.4) + 0.5 * np.sin(2 * np.pi * f * np.arange(n) / SR)
    cut = 180 + 1500 * np.exp(-np.arange(n) / (0.07 * SR))
    y = svf_lp(x, cut, 0.55)
    a = np.minimum(1, np.arange(n) / (0.004 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.01 * SR))
    return y * a * vel

def stab(notes, d=0.16, vel=1.0):
    n = int(d * SR); y = np.zeros(n)
    for m in notes:
        f = mtof(m)
        y += saw(f, n, 0.004) + saw(f, n, -0.004) + 0.6 * sqr(f * 2, n, 0.3)
    cut = 500 + 4200 * np.exp(-np.arange(n) / (0.045 * SR))
    y = svf_lp(y, cut, 0.8)
    a = np.minimum(1, np.arange(n) / (0.002 * SR)) * np.exp(-np.arange(n) / (0.09 * SR))
    return y * a * vel / len(notes) * 1.8

def pluck(m, d=0.22, vel=1.0):
    n = int(d * SR); f = mtof(m)
    y = sqr(f, n, 0.25) * 0.7 + saw(f * 1.003, n) * 0.5
    cut = 900 + 5000 * np.exp(-np.arange(n) / (0.05 * SR))
    y = svf_lp(y, cut, 0.7)
    return y * np.exp(-np.arange(n) / (0.11 * SR)) * vel


def riser(d, f0=300, f1=9000):
    n = int(d * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    cut = f0 * (f1 / f0) ** (t / d)
    y = svf_lp(nz, cut * 0.9, 0.35) - svf_lp(nz, cut * 0.25, 0.9)
    return y * (t / d) ** 1.6


# ------------------------------------------------------------------ arrangement
# chords (midi) -- Am9 / D9 / E7#9 funk loop
AM9 = [57, 60, 64, 67, 71]; D9 = [57, 60, 62, 66, 64]; E7 = [56, 59, 62, 66, 63]; FM7 = [57, 60, 65, 64]
BASS = {
    'A': [(0, 33, 2), (3, 33, 1), (6, 45, 1), (8, 40, 2), (10, 43, 1), (12, 33, 1), (14, 36, 1), (15, 38, 1)],
    'D': [(0, 38, 2), (3, 38, 1), (6, 50, 1), (8, 45, 2), (10, 48, 1), (12, 38, 1), (14, 42, 1), (15, 43, 1)],
    'E': [(0, 40, 2), (3, 40, 1), (6, 52, 1), (8, 47, 2), (10, 50, 1), (12, 40, 1), (14, 44, 1), (15, 45, 1)],
}
# bar index (0..7) -> (bass key, chord)
PROG = {0: ('A', AM9), 1: ('A', AM9), 2: ('D', D9), 3: ('E', E7),
        4: ('A', AM9), 5: ('D', D9), 6: ('A', AM9), 7: ('E', E7)}
LEAD = [  # (bar, step, midi, len) -- bright syncopated hook, bars 3-7
    (3, 0, 76, 2), (3, 3, 79, 1), (3, 6, 76, 1), (3, 8, 74, 2), (3, 11, 72, 1), (3, 14, 76, 1),
    (4, 0, 81, 2), (4, 3, 79, 1), (4, 6, 76, 1), (4, 8, 79, 2), (4, 11, 76, 1), (4, 14, 72, 1),
    (5, 0, 74, 2), (5, 3, 76, 1), (5, 6, 79, 1), (5, 8, 81, 2), (5, 11, 79, 1), (5, 14, 76, 1),
    (6, 0, 81, 2), (6, 3, 84, 1), (6, 6, 81, 1), (6, 8, 79, 2), (6, 11, 76, 1), (6, 14, 79, 1),
    (7, 0, 84, 2), (7, 3, 81, 1), (7, 6, 79, 1), (7, 8, 76, 2), (7, 11, 79, 1), (7, 14, 81, 1),
]


def make_music():
    buf = np.zeros((N, 2)); drums = np.zeros((N, 2)); pitched = np.zeros((N, 2))
    kick_times = []
    for bar in range(8):
        t0 = bar * 4 * BEAT
        bk, chord = PROG[bar]
        energy = 0.8 if bar < 2 else 1.0
        # --- drums
        for beat in range(4):
            tb = t0 + beat * BEAT
            if bar == 7 and beat == 3: continue  # last beat handled by the ending hits
            place(drums, KICK, tb, 1.0 * energy)
            kick_times.append(tb)
        if bar >= 1:
            place(drums, KICK, t0 + 10 * S16, 0.55)  # ghost kick
        for beat in (1, 3):
            if bar >= 1:
                place(drums, CLAP, t0 + beat * BEAT, 0.8)
                place(drums, SNARE, t0 + beat * BEAT, 0.35)
        for s in range(16):
            ts = t0 + s * S16
            vel = 0.32 if s % 4 == 0 else (0.22 if s % 2 == 0 else 0.14)
            if s % 4 == 2: place(drums, HAT_O, ts, 0.28, 0.15)  # open hat on the offbeat
            else: place(drums, HAT_C, ts, vel, -0.15 if s % 2 else 0.15)
            if bar >= 2 and s % 2 == 1: place(drums, SHK, ts, 0.18, 0.3)
        # --- fills / builds
        if bar == 2:  # 4..6s : snare roll into the big reveal
            for s in range(8, 16):
                place(drums, SNARE, t0 + s * S16, 0.18 + 0.05 * (s - 8))
            place(drums, riser(2.0, 250, 9000), t0, 0.35)
        if bar == 5:  # 10..12s : build into the closer
            for s in range(12, 16):
                place(drums, SNARE, t0 + s * S16, 0.35 + 0.1 * (s - 12))
            place(drums, riser(1.5, 300, 8000), t0 + 0.5, 0.3)
        # --- bass
        for (s, m, ln) in BASS[bk]:
            if bar == 2 and s >= 14: continue
            place(pitched, bass_note(m, ln * S16 * 0.95, 0.9), t0 + s * S16, 0.9)
        # --- chord stabs (off-beat funk)
        stab_steps = [3, 6, 10, 13] if bar % 2 == 0 else [3, 5, 8, 11, 14]
        for s in stab_steps:
            place(pitched, stab(chord, 0.14, 0.9), t0 + s * S16, 0.42, 0.25 if s % 2 else -0.25)
        # --- pad-ish sustained high note bed (adds shine) from bar 3
    for (bar, s, m, ln) in LEAD:
        t = bar * 4 * BEAT + s * S16
        if bar == 7 and s >= 11: continue
        pl = pluck(m, 0.24, 0.9)
        place(pitched, pl, t, 0.28, 0.1)
        place(pitched, pl, t + 3 * S16, 0.10, -0.5)   # dotted-8th echo
    # impacts / crashes on section hits
    for t, g in ((6.0, 1.0), (12.0, 0.9), (15.0, 1.0)):
        place(drums, crash(1.8), t, 0.5 * g)
        place(pitched, stab(AM9 if t != 12.0 else AM9, 0.5, 1.0), t, 0.5)
    # ending: stop-time hits on 15.5 & 15.75, hard stop at 16.0
    for t in (15.5, 15.75):
        place(drums, KICK, t, 1.0); place(drums, CLAP, t, 0.9)
        place(pitched, stab(AM9, 0.2, 1.0), t, 0.6)
    place(drums, crash(0.5), 15.75, 0.5)

    # duck the pitched material under the kick (sidechain pump)
    duck = np.ones(N)
    for tk in kick_times + [15.5, 15.75]:
        i = int(tk * SR); m = int(0.22 * SR)
        if i + m > N: m = N - i
        duck[i:i + m] = np.minimum(duck[i:i + m], 1 - 0.55 * np.exp(-np.arange(m) / (0.07 * SR)))
    pitched *= duck[:, None]
    mix = drums * 0.95 + pitched * 0.85
    # intro: filter-in feel (first 1.0s quieter highs) is not needed; just a soft fade-in of 80 ms
    mix[: int(0.08 * SR)] *= np.linspace(0, 1, int(0.08 * SR))[:, None]
    # hard-ish stop with 60 ms fade
    k = int(0.06 * SR); mix[-k:] *= np.linspace(1, 0, k)[:, None]
    return mix


# ------------------------------------------------------------------ SFX
def sfx_pop(f0=520, f1=180, d=0.14):
    n = int(d * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.05)
    y += hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.006) * 0.35
    return y
def sfx_bloop(f0=300, f1=900, d=0.16):
    n = int(d * SR); t = np.arange(n) / SR
    f = f0 + (f1 - f0) * (1 - np.exp(-t / 0.035))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)
def sfx_whoosh(d, peak, f0=250, f1=7000, boom=True):
    n = int(d * SR); t = np.arange(n) / SR
    nz = rng.standard_normal(n)
    sweep = f0 * (f1 / f0) ** np.clip(t / (peak * d), 0, 1) * (1 - 0.6 * np.clip((t - peak * d) / (d * (1 - peak) + 1e-9), 0, 1))
    y = svf_lp(nz, sweep, 0.45) - 0.5 * svf_lp(nz, sweep * 0.3, 0.6)
    a = np.where(t < peak * d, (t / (peak * d)) ** 2.0, np.exp(-(t - peak * d) / (0.12 * d + 1e-9)))
    y = y * a
    return y / (np.max(np.abs(y)) + 1e-9)
def sfx_thump(d=0.3):
    n = int(d * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * np.cumsum(55 + 60 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.12)
def sfx_sparkle(d=0.5):
    n = int(d * SR); y = np.zeros(n)
    for k, f in enumerate([1568, 2093, 2637, 3136, 4186]):
        i = int(k * 0.04 * SR)
        m = n - i; t = np.arange(m) / SR
        y[i:] += np.sin(2 * np.pi * f * t) * np.exp(-t / 0.08) * 0.3
    return y

def make_sfx():
    buf = np.zeros((N, 2))
    for t, kind in SFX:
        if kind == "pop": place(buf, sfx_pop(), t, 0.9)
        elif kind == "pop2": place(buf, sfx_pop(620, 220), t, 0.9)
        elif kind == "pop3": place(buf, sfx_pop(740, 260), t, 0.9)
        elif kind == "pop_big":
            place(buf, sfx_pop(700, 120, 0.22), t, 1.1); place(buf, sfx_sparkle(), t + 0.02, 0.5)
            place(buf, sfx_thump(0.4), t, 0.5)
        elif kind == "unpop": place(buf, sfx_bloop(900, 250, 0.12), t, 0.5)
        elif kind == "unpop2": place(buf, sfx_bloop(700, 200, 0.1), t + 0.03, 0.35)
        elif kind == "swish": place(buf, sfx_whoosh(0.3, 0.8, 800, 6000), t, 0.5, 0.3)
        elif kind == "whoosh_big":
            w = sfx_whoosh(0.75, 0.33, 200, 9000)  # swell peaks 0.25s after start => at 6.0
            place(buf, w, t, 1.1, -0.2); place(buf, w[::-1][:int(0.2 * SR)] * 0.0, t, 0)
            place(buf, sfx_thump(0.6), 6.0, 0.9)
        elif kind == "whoosh_fast":
            w = sfx_whoosh(0.5, 0.35, 300, 8000)  # peak ~0.17s => 11.92
            place(buf, w, t, 0.95, 0.2)
        elif kind == "thump": place(buf, sfx_thump(), t, 0.55)
        elif kind == "impact":
            place(buf, sfx_thump(0.9), t, 0.9); place(buf, crash(1.0), t, 0.35)
    return buf


if __name__ == "__main__":
    m = make_music(); s = make_sfx()
    wavfile.write("music.wav", SR, (np.clip(m / np.max(np.abs(m)) * 0.9, -1, 1) * 32767).astype(np.int16))
    wavfile.write("sfx.wav", SR, (np.clip(s / np.max(np.abs(s)) * 0.9, -1, 1) * 32767).astype(np.int16))
    print("ok", m.shape, np.abs(m).max(), np.abs(s).max())
