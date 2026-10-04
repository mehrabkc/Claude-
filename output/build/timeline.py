"""Single source of truth for timing. 120 BPM -> beat = 0.5s, bar = 2s, so every
pointer / punch time in the brief (4, 6, 7, 10, 12, 14, 15) sits exactly on a beat."""
BPM = 120.0
BEAT = 60.0 / BPM
DUR = 16.0

# source cut points (verified with scene detection)
CUT1 = 6.0          # clip A -> B
CUT2 = 11.917       # clip B -> C
SRC_END = 15.9      # CapCut end card begins here, never used

# whip pans: (start, end, direction)  cut sits at the midpoint of each window
WHIP1 = (5.75, 6.25, -1)   # the BIG reveal whip
WHIP2 = (11.75, 12.083, +1)  # fast whip into the closer

# overlay windows
T_1500_IN, T_1500_OUT = 4.0, 6.85
T_ARROW_IN = 5.72
T_440_IN, T_440_OUT = 6.0, 6.88          # logo window 7.0-8.0 stays completely clean
STEP_IN = (10.0, 10.5, 11.0)
STEPS_OUT = 11.58                         # all gone before WHIP2 starts at 11.75
T_CTA1, T_CTA2, T_CTA_END = 14.0, 15.0, 16.0

# hard punch-in zooms requested in the brief
PUNCHES = (4.0, 6.0, 7.0, 10.0)

# sfx events: (time, kind)
SFX = [
    (4.0, "pop"),
    (5.70, "swish"),            # arrow draws on
    (5.75, "whoosh_big"),       # peaks at 6.0 (cut)
    (6.0, "pop_big"),
    (6.88, "unpop"), (6.88, "unpop2"),
    (7.0, "thump"),
    (10.0, "pop"), (10.5, "pop2"), (11.0, "pop3"),
    (10.0, "thump"),
    (11.58, "unpop"),
    (11.75, "whoosh_fast"),     # peaks near 11.92
    (12.0, "thump"),
    (14.0, "pop_big"),
    (15.0, "pop_big"),
    (15.0, "impact"),
]
