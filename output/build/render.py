import cv2, numpy as np, subprocess, json, math, sys, random
from assets import *
from timeline import *

SRC = "/root/.claude/uploads/28cefa92-73e1-5625-b696-5063f9b1eaf4/ed0a8c14-copy_CED29BB7-182F-40F1-ABE0-8B69F31CDEFD.mov"
W, H = 1080, 1920
FPS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] in ("full",) else 30
NF = int(DUR * FPS)

# ------------------------------------------------------------------ source + face track
def load_src(fps):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", SRC, "-t", str(SRC_END), "-vf", f"fps={fps},format=rgb24",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    raw = p.stdout.read()
    n = len(raw) // (W * H * 3)
    return np.frombuffer(raw, np.uint8).reshape(n, H, W, 3)

SRCF = load_src(FPS)
NS = len(SRCF)
C1 = int(round(CUT1 * FPS)); C2 = int(round(CUT2 * FPS))   # first frame idx of B and C
faces30 = np.array(json.load(open("faces.json")))
def face_at(idx):  # normalised face centre for a source frame index (faces were tracked at 30fps)
    j = min(len(faces30) - 1, int(round(idx * 30 / FPS)))
    return faces30[j]

# clip frame access: returns (frame, face) for clip k at time t, clamped inside the clip
CLIPS = [(0, C1 - 1), (C1, C2 - 1), (C2, NS - 1)]
def clip_frame(k, t):
    lo, hi = CLIPS[k]
    idx = int(min(hi, max(lo, math.floor(t * FPS + 1e-6))))
    return SRCF[idx], face_at(idx)

# ------------------------------------------------------------------ easing
def clamp01(x): return max(0.0, min(1.0, x))
def ease_out_cubic(p): return 1 - (1 - p) ** 3
def ease_out_back(p, s=2.2):
    p = clamp01(p) - 1; return 1 + p * p * ((s + 1) * p + s)
def ease_in_back(p, s=1.8):
    p = clamp01(p); return p * p * ((s + 1) * p - s)
def ease_io_quint(p):
    p = clamp01(p); return 16 * p ** 5 if p < .5 else 1 - (-2 * p + 2) ** 5 / 2
def lerp(a, b, p): return a + (b - a) * p

# ------------------------------------------------------------------ zoom plan: (t0, z_at_start, z_at_end_of_segment)
ZOOM = [
    (0.0, 1.00, 1.04), (1.0, 1.09, 1.06), (1.5, 1.01, 1.07), (2.5, 1.12, 1.09), (3.0, 1.02, 1.10),
    (4.0, 1.20, 1.17),   # HARD PUNCH  (৳1500 kinlei)
    (5.0, 1.04, 1.14),
    (6.0, 1.32, 1.26),   # HARD PUNCH  (৳440 OFF! lands)
    (7.0, 1.42, 1.30),   # HARD PUNCH  (brand name)
    (8.0, 1.06, 1.16), (9.0, 1.26, 1.20), (9.5, 1.08, 1.18),
    (10.0, 1.36, 1.30),  # HARD PUNCH  (steps start)
    (11.0, 1.12, 1.22),
    (12.0, 1.30, 1.22), (12.5, 1.04, 1.14), (13.0, 1.22, 1.28), (13.5, 1.08, 1.18),
    (14.0, 1.30, 1.25),  # CTA #1
    (15.0, 1.36, 1.44),  # CTA #2 -- final impact push
    (16.5, 1.5, 1.5),
]
SNAP = 0.085   # seconds for a "hard" zoom snap
def zoom_at(t):
    for i in range(len(ZOOM) - 1):
        t0, zs, ze = ZOOM[i]; t1 = ZOOM[i + 1][0]
        if t0 <= t < t1:
            z = lerp(zs, ze, (t - t0) / (t1 - t0))
            if i > 0 and t - t0 < SNAP:
                prev_end = ZOOM[i - 1][2]
                z = lerp(prev_end, z, ease_out_cubic((t - t0) / SNAP))
            return z, t0
    return ZOOM[-1][1], ZOOM[-1][0]

def beat_pulse(t):
    ph = t % BEAT
    return 0.022 * math.exp(-ph / 0.09)

def tilt_at(t):
    z, t0 = zoom_at(t)
    k = [i for i, s in enumerate(ZOOM) if s[0] == t0][0]
    amp = 2.0 if t0 in PUNCHES else 1.3
    return amp * (1 if k % 2 else -1) * math.exp(-(t - t0) / 0.2)

def shake_at(t):
    """screen shake on the two big hits"""
    s = 0.0
    for th, a in ((6.0, 22), (15.0, 18), (7.0, 8), (4.0, 8), (10.0, 8)):
        d = t - th
        if 0 <= d < 0.35:
            s = max(s, a * math.exp(-d / 0.09))
    return s

# ------------------------------------------------------------------ view warp
def crop_view(src, face, z, rot, tx=0.0, ty=0.0, border=cv2.BORDER_REFLECT_101):
    w, h = W / z, H / z
    fx, fy = face[0] * W, face[1] * H
    left = min(max(fx - 0.5 * w, 0), W - w)
    top = min(max(fy - 0.30 * h, 0), H - h)
    cx, cy = left + w / 2, top + h / 2
    M = cv2.getRotationMatrix2D((cx, cy), rot, z)
    M[0, 2] += W / 2 - cx + tx; M[1, 2] += H / 2 - cy + ty
    return cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border), M

def view_at(t, subt, i_cur):
    """Rendered view at (sub)time subt. Handles whip pans via a two-clip strip."""
    z0, _ = zoom_at(subt)
    z = z0 * (1 + beat_pulse(subt))
    rot = tilt_at(subt)
    sh = shake_at(subt)
    sx = sh * math.sin(subt * 130) ; sy = sh * math.cos(subt * 110)
    for (w0, w1, d), (ka, kb) in ((WHIP1, (0, 1)), (WHIP2, (1, 2))):
        if w0 <= subt <= w1:
            p = (subt - w0) / (w1 - w0)
            e = ease_io_quint(p)
            zz = z * (1 + 0.10 * math.sin(math.pi * p))
            fa_src, fa = clip_frame(ka, subt); fb_src, fb = clip_frame(kb, subt)
            txa = d * W * e; txb = txa - d * W
            va, _ = crop_view(fa_src, fa, zz, rot, txa + sx, sy, cv2.BORDER_CONSTANT)
            vb, _ = crop_view(fb_src, fb, zz, rot, txb + sx, sy, cv2.BORDER_CONSTANT)
            # coverage masks from a white image (A and B tile edge to edge)
            ma = np.zeros((H, W), np.uint8); xa = int(round(txa))
            if d < 0: ma[:, :max(0, W + xa)] = 1
            else: ma[:, min(W, xa):] = 1
            out = np.where(ma[..., None] == 1, va, vb)
            return out, p
    k = 0 if subt < CUT1 else (1 if subt < CUT2 else 2)
    f, fc = clip_frame(k, subt)
    v, _ = crop_view(f, fc, z, rot, sx, sy)
    return v, None

def whip_progress(t):
    for (w0, w1, d) in (WHIP1, WHIP2):
        if w0 <= t <= w1: return (t - w0) / (w1 - w0)
    return None

def render_video(t, i):
    # motion-blur sub-sampling: more samples on whip frames and just after hard snaps
    in_whip = whip_progress(t) is not None
    snap_recent = any(0 <= t - s[0] < 0.11 for s in ZOOM[1:])
    if in_whip: K, shut = 18, 1.6 / FPS
    elif snap_recent: K, shut = 5, 0.8 / FPS
    else: K, shut = 1, 0.0
    if K == 1:
        v, p = view_at(t, t, i); frame = v
    else:
        acc = np.zeros((H, W, 3), np.float32)
        for k in range(K):
            ts = t - shut * (1 - k / (K - 1))
            v, _ = view_at(t, max(ts, 0), i)
            acc += v
        frame = (acc / K).astype(np.uint8)
    p = whip_progress(t)
    if p is not None:
        speed = math.sin(math.pi * p) ** 1.5
        big = WHIP1[0] <= t <= WHIP1[1]
        kx = int(speed * (200 if big else 130)) | 1
        if kx > 3: frame = cv2.blur(frame, (kx, 1))
        # speed lines
        lay = np.zeros((H, W, 3), np.uint8)
        rng = np.random.default_rng(i)
        for _ in range(int(36 * speed)):
            y = int(rng.integers(120, H - 120)); x0 = int(rng.integers(-200, W)); ln = int(rng.integers(300, 900))
            cv2.line(lay, (x0, y), (x0 + ln, y), (255, 255, 255), int(rng.integers(2, 6)), cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (31, 1), 0)
        frame = cv2.addWeighted(frame, 1.0, lay, 0.55 * speed, 0)
    return frame

def grade(frame, t):
    f = frame.astype(np.float32)
    y = (f * np.array([0.299, 0.587, 0.114], np.float32)).sum(2, keepdims=True)
    f = y + (f - y) * 1.10
    f = (f - 128) * 1.03 + 128
    return np.clip(f, 0, 255).astype(np.uint8)

def chroma(frame, amt):
    amt = int(amt)
    if amt < 1: return frame
    out = frame.copy()
    out[:, amt:, 0] = frame[:, :-amt, 0]       # red channel pushed right
    out[:, :-amt, 2] = frame[:, amt:, 2]       # blue channel pushed left
    return out

# ------------------------------------------------------------------ overlay sprites
def T(pm): return to_sprite(pm)
SP = {}
SP["t1500"] = T(text_pm(["৳1500 kinlei"], YEL, 124))
SP["bag"] = T(badge_with_icon(CYAN, icon_bag, 178, dots=(0, 205, 196)))
SP["t440"] = T(text_pm(["৳440 OFF!"], PINK, 150))
SP["burst440"] = T(burst_sprite(445, 272, 22, 0.82, YEL, ORANGE, wob=0.05, seed=3))
SP["tag"] = T(badge_with_icon(CYAN, icon_tag, 172, rot=-12, dots=(0, 205, 196)))
STEP_FONT = 80
def step_sprite(label, color, icon_fn, dots):
    b = badge_with_icon(color, icon_fn, 150, dots=dots)
    tx = text_pm([label], color, STEP_FONT)
    return T(hcompose([b, tx], gap=2))
SP["s1"] = step_sprite("Kino", CYAN, icon_bag, (0, 205, 196))
SP["s2"] = step_sprite("Post koro", YEL, icon_phone, (215, 210, 0))
SP["s3"] = step_sprite("Receipt pathao", PINK, icon_receipt, (214, 0, 144))
SP["cta_burst"] = T(burst_sprite(470, 268, 20, 0.84, CYAN, (0, 196, 186), wob=0.06, seed=11, outline=12, border=8))
SP["cta1a"] = T(text_pm(["Offer ta nao"], YEL, 124, stroke=0.12))
SP["cta1b"] = T(text_pm(["ekhoni!"], YEL, 170, stroke=0.12))
SP["cta2a"] = T(text_pm(["Life level"], PINK, 134, stroke=0.12))
SP["cta2b"] = T(text_pm(["up koro!"], PINK, 170, stroke=0.12))

def blit(frame, spr, cx, cy, scale, rot=0.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0: return
    h, w = spr.shape[:2]; s = scale / SSF
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -rot, s)
    M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    pts = M @ c
    x0 = int(max(0, math.floor(pts[0].min()))); x1 = int(min(W, math.ceil(pts[0].max())))
    y0 = int(max(0, math.floor(pts[1].min()))); y1 = int(min(H, math.ceil(pts[1].max())))
    if x1 <= x0 or y1 <= y0: return
    M[0, 2] -= x0; M[1, 2] -= y0
    wp = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = wp[..., 3:4] * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1 - a) + wp[..., :3] * alpha, 0, 255).astype(np.uint8)

POP_D = 0.26; OUT_D = 0.14
def anim(t, t_in, t_out, wob=9.0, seed=0):
    """-> (scale, rot_offset, alpha) pop-in with overshoot + wobble, quick pop-out, beat pulse + sway while idle."""
    if t < t_in: return 0.0, 0.0, 0.0
    p = (t - t_in) / POP_D
    s = ease_out_back(p) if p < 1 else 1.0
    rot = wob * (1 - clamp01(p)) * math.cos(clamp01(p) * math.pi * 2.5) * (1 if seed % 2 else -1)
    if p >= 1:
        ph = (t - t_in) % BEAT
        s *= 1 + 0.035 * math.exp(-ph / 0.10)
        rot += 1.4 * math.sin((t - t_in) * 2 * math.pi / (2 * BEAT) + seed)
    if t_out is not None and t > t_out - OUT_D:
        q = clamp01((t - (t_out - OUT_D)) / OUT_D)
        s *= max(0.0, 1 - ease_in_back(q) * 1.0) if q < 1 else 0.0
        if q >= 1: return 0.0, 0.0, 0.0
    return max(s, 0.0), rot, 1.0

# --- particles / rays
def star_pts(cx, cy, r, rot):
    pts = []
    for k in range(8):
        a = rot + k * math.pi / 4; rr = r if k % 2 == 0 else r * 0.38
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return np.array(pts, np.int32).reshape(-1, 1, 2)

def particles(frame, t, t0, cx, cy, n, colors, speed=520, life=0.55, size=26, seed=1, rays=True, ring=True):
    d = t - t0
    if d < 0 or d > life: return
    rng = np.random.default_rng(seed)
    q = d / life
    if ring and q < 0.55:
        r = int(60 + 380 * ease_out_cubic(q / 0.55)); th = max(1, int(16 * (1 - q / 0.55)))
        ov = frame.copy(); cv2.circle(ov, (int(cx), int(cy)), r, (255, 255, 255), th + 6, cv2.LINE_AA)
        cv2.circle(ov, (int(cx), int(cy)), r, (255, 255, 255), th, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(frame, 1 - 0.7 * (1 - q / 0.55), ov, 0.7 * (1 - q / 0.55), 0)
    for k in range(n):
        ang = rng.uniform(0, 2 * math.pi); sp = rng.uniform(0.45, 1.0) * speed
        r = sp * ease_out_cubic(q) * life
        x = cx + math.cos(ang) * r; y = cy + math.sin(ang) * r * 0.9 + 160 * q * q
        col = colors[k % len(colors)]; sz = size * rng.uniform(0.6, 1.2) * (1 - q ** 1.5)
        if sz < 2: continue
        shape = k % 3
        if shape == 0:
            cv2.fillPoly(frame, [star_pts(x, y, sz * 1.5 + 5, t * 6 + k)], BLACK, cv2.LINE_AA)
            cv2.fillPoly(frame, [star_pts(x, y, sz * 1.5, t * 6 + k)], col, cv2.LINE_AA)
        elif shape == 1:
            cv2.circle(frame, (int(x), int(y)), int(sz * 0.7) + 4, BLACK, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), int(sz * 0.7), col, -1, cv2.LINE_AA)
        else:
            a2 = t * 8 + k; dx, dy = math.cos(a2) * sz, math.sin(a2) * sz
            p1 = (int(x - dx), int(y - dy)); p2 = (int(x + dx), int(y + dy))
            cv2.line(frame, p1, p2, BLACK, 14, cv2.LINE_AA); cv2.line(frame, p1, p2, col, 8, cv2.LINE_AA)

# --- arrows
def bez(p0, p1, p2, p3, n=60):
    ts = np.linspace(0, 1, n)[:, None]
    return ((1 - ts) ** 3 * np.array(p0) + 3 * (1 - ts) ** 2 * ts * np.array(p1) + 3 * (1 - ts) * ts ** 2 * np.array(p2) + ts ** 3 * np.array(p3))

def draw_arrow(frame, ctrl, prog, color, w=30, out_w=15, alpha=1.0, head=1.0):
    if prog <= 0: return
    pts = bez(*ctrl); n = len(pts); m = max(2, int(n * clamp01(prog)))
    seg = pts[:m].astype(np.float32)
    shadow = (seg + np.array([8, 12], np.float32)).astype(np.int32).reshape(-1, 1, 2)
    ov2 = frame.copy()
    cv2.polylines(ov2, [shadow], False, (0, 0, 0), w + 2 * out_w + 6, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(frame, 0.6, ov2, 0.4, 0)
    sp = seg.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [sp], False, WHITE, w + 2 * out_w + 10, cv2.LINE_AA)
    cv2.polylines(frame, [sp], False, BLACK, w + 2 * out_w, cv2.LINE_AA)
    # arrowhead once the shaft is nearly there
    if prog > 0.8:
        hp = clamp01((prog - 0.8) / 0.2)
        tip = pts[-1]; back = pts[-6]; v = tip - back; v /= (np.linalg.norm(v) + 1e-6)
        nrm = np.array([-v[1], v[0]]); L = 110 * head * ease_out_back(hp, 1.4); Wd = 78 * head * ease_out_back(hp, 1.4)
        tri = lambda grow: np.array([tip + v * (L * 0.55 + grow), tip - v * (L * 0.45) + nrm * (Wd + grow), tip - v * (L * 0.45) - nrm * (Wd + grow)], np.int32)
        cv2.fillPoly(frame, [tri(18)], WHITE, cv2.LINE_AA)
        cv2.fillPoly(frame, [tri(10)], BLACK, cv2.LINE_AA)
        cv2.fillPoly(frame, [tri(0)], color, cv2.LINE_AA)
    cv2.polylines(frame, [sp], False, color, w, cv2.LINE_AA)


# ------------------------------------------------------------------ overlay timeline
POS = dict(t1500=(565, 990), bag=(108, 862), burst440=(540, 1350), t440=(540, 1350), tag=(190, 1160),
           cta=(540, 1325))
def _w(k): return SP[k].shape[1] / SSF
POS["s1"] = (60 + _w("s1") / 2, 1055); POS["s2"] = (150 + _w("s2") / 2, 1290); POS["s3"] = (60 + _w("s3") / 2, 1525)
ARROW_MAIN = [(935, 975), (1045, 1030), (1035, 1135), (905, 1160)]
_b1 = 60 + 75; _b2 = 150 + 75
ARROW_P1 = [(_b1 - 5, 1140), (_b1 - 10, 1165), (_b2 - 30, 1170), (_b2 - 25, 1195)]
ARROW_P2 = [(_b2 - 5, 1375), (_b2 - 10, 1400), (_b1 + 30, 1405), (_b1 + 20, 1430)]

def draw_overlay(frame, t):
    # ---- ৳1500 kinlei + bag (4.0 -> 6.85)
    s, r, a = anim(t, T_1500_IN, T_440_OUT - 0.02, seed=1)
    if a:
        blit(frame, SP["t1500"], *POS["t1500"], s, -4 + r)
        sb, rb, ab = anim(t, T_1500_IN + 0.06, T_440_OUT - 0.02, wob=18, seed=2)
        blit(frame, SP["bag"], *POS["bag"], sb, -10 + rb)
    # ---- arrow 1500 -> 440
    if T_ARROW_IN <= t < T_440_OUT:
        prog = ease_out_cubic((t - T_ARROW_IN) / 0.28)
        fade = 1.0
        if t > T_440_OUT - OUT_D:
            fade = max(0.0, (T_440_OUT - t) / OUT_D); prog = min(prog, fade)
        draw_arrow(frame, ARROW_MAIN, prog, CYAN)
    # ---- ৳440 OFF! burst
    s, r, a = anim(t, T_440_IN, T_440_OUT, wob=14, seed=3)
    if a:
        cx, cy = POS["burst440"]
        blit(frame, SP["burst440"], cx, cy, s * 1.03, 2 + r * 0.7)
        blit(frame, SP["t440"], cx, cy + 4, s, 2 + r)
        st, rt, at = anim(t, T_440_IN + 0.07, T_440_OUT, wob=20, seed=4)
        blit(frame, SP["tag"], *POS["tag"], st, -12 + rt)
    particles(frame, t, T_440_IN, 540, 1350, 16, [YEL, PINK, CYAN, WHITE, ORANGE], speed=620, life=0.6, size=30, seed=5)
    particles(frame, t, T_1500_IN, 540, 990, 9, [YEL, CYAN, PINK], speed=420, life=0.5, size=22, seed=6, ring=False)

    # ---- 3-step process (10.0 -> 11.58)
    for k, key in enumerate(("s1", "s2", "s3")):
        s, r, a = anim(t, STEP_IN[k], STEPS_OUT + 0.03 * k, wob=12, seed=k + 1)
        if a: blit(frame, SP[key], *POS[key], s, (-2, 2, -2)[k] + r)
    if STEP_IN[1] <= t < STEPS_OUT:
        pr = ease_out_cubic((t - STEP_IN[1] + 0.0) / 0.25)
        if t > STEPS_OUT - OUT_D: pr = min(pr, (STEPS_OUT - t) / OUT_D)
        draw_arrow(frame, ARROW_P1, pr, LIME, w=14, out_w=8, head=0.55)
    if STEP_IN[2] <= t < STEPS_OUT:
        pr = ease_out_cubic((t - STEP_IN[2]) / 0.25)
        if t > STEPS_OUT - OUT_D: pr = min(pr, (STEPS_OUT - t) / OUT_D)
        draw_arrow(frame, ARROW_P2, pr, LIME, w=14, out_w=8, head=0.55)
    for k, (cx, cy) in enumerate((POS["s1"], POS["s2"], POS["s3"])):
        particles(frame, t, STEP_IN[k], cx - 60, cy, 8, [CYAN, YEL, PINK, WHITE], speed=330, life=0.45, size=18, seed=20 + k, ring=False)

    # ---- CTA (14.0 -> end)
    cx, cy = POS["cta"]
    sb, rb, ab = anim(t, T_CTA1, None, wob=10, seed=5)
    if ab:
        if t >= T_CTA2:  # burst gets a jolt on the swap
            sb *= 1 + 0.10 * math.exp(-(t - T_CTA2) / 0.12)
        blit(frame, SP["cta_burst"], cx, cy, sb, -2 + rb * 0.6)
    if T_CTA1 <= t < T_CTA2:
        s1, r1, a1 = anim(t, T_CTA1 + 0.05, T_CTA2 - 0.02, wob=7, seed=1)
        s2, r2, a2 = anim(t, T_CTA1 + 0.2, T_CTA2 - 0.02, wob=10, seed=2)
        if a1: blit(frame, SP["cta1a"], cx, cy - 70, s1, -3 + r1)
        if a2: blit(frame, SP["cta1b"], cx, cy + 78, s2, -3 + r2)
    if t >= T_CTA2:
        s1, r1, a1 = anim(t, T_CTA2 + 0.0, None, wob=8, seed=2)
        s2, r2, a2 = anim(t, T_CTA2 + 0.13, None, wob=10, seed=1)
        if a1: blit(frame, SP["cta2a"], cx, cy - 70, s1, 3 + r1)
        if a2: blit(frame, SP["cta2b"], cx, cy + 78, s2, 3 + r2)
    particles(frame, t, T_CTA1, cx, cy, 14, [YEL, PINK, WHITE, ORANGE, LIME], speed=640, life=0.6, size=28, seed=31)
    particles(frame, t, T_CTA2, cx, cy, 22, [PINK, YEL, CYAN, WHITE, LIME, PURPLE], speed=760, life=0.7, size=32, seed=32)


def compose(i):
    t = i / FPS
    frame = render_video(t, i)
    frame = grade(frame, t)
    # chromatic punch on whip peaks + hard hits
    amt = 0
    for th, a in ((6.0, 14), (11.917, 9), (4.0, 5), (7.0, 6), (10.0, 5), (15.0, 8)):
        d = abs(t - th)
        if d < 0.12: amt = max(amt, a * (1 - d / 0.12))
    frame = chroma(frame, amt)
    # white flash on the cut peaks
    for th, a in ((6.0, 0.45), (11.917, 0.25), (15.0, 0.18)):
        d = abs(t - th)
        if d < 0.07:
            frame = cv2.addWeighted(frame, 1 - a * (1 - d / 0.07), np.full_like(frame, 255), a * (1 - d / 0.07), 0)
    draw_overlay(frame, t)
    return frame


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "preview":
        times = [float(x) for x in sys.argv[2].split(",")]
        tiles = []
        for t in times:
            f = compose(int(round(t * FPS)))
            tiles.append(cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA))
        rows = [np.hstack(tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6]))) for i in range(0, len(tiles), 6)]
        cv2.imwrite(sys.argv[3], cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    elif mode == "full":
        out = sys.argv[3] if len(sys.argv) > 3 else "silent.mp4"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                                "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        for i in range(NF):
            enc.stdin.write(compose(i).tobytes())
            if i % 60 == 0: print(i, "/", NF, flush=True)
        enc.stdin.close(); enc.wait()
