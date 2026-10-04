"""Sprite factory: Flex pop-culture stickers, outlined text, icons. Everything is built at RS x then
resampled to SSF x design size and stored as premultiplied float32 RGBA."""
import cv2, numpy as np, math
from PIL import Image, ImageDraw, ImageFont

PINK = (255, 4, 172); YEL = (248, 244, 23); CYAN = (0, 255, 240)
ORANGE = (247, 147, 1); PURPLE = (175, 49, 170); LIME = (41, 199, 48)
WHITE = (255, 255, 255); BLACK = (0, 0, 0)

RS = 2.0      # internal render scale
SSF = 1.25    # stored scale relative to design size
TITAN = "fonts/TitanOne-Regular.ttf"
BALOO = "fonts/BalooDa2[wght].ttf"


def _font(path, size, var=None):
    f = ImageFont.truetype(path, int(round(size)))
    if var: f.set_variation_by_name(var)
    return f


def finalize(rgba_u8):
    """uint8 RGBA (straight alpha, built at RS) -> premultiplied float32 at SSF."""
    h, w = rgba_u8.shape[:2]
    a = rgba_u8[..., 3:].astype(np.float32) / 255
    pm = np.concatenate([rgba_u8[..., :3].astype(np.float32) * a, a], axis=2)
    f = SSF / RS
    pm = cv2.resize(pm, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return pm


# ---------------------------------------------------------------- compositing helpers (premultiplied)
def _over(dst, src):
    return src + dst * (1 - src[..., 3:])

def _layer(mask, color, alpha=1.0):
    m = (mask.astype(np.float32) / 255 * alpha)[..., None]
    c = np.array(color, np.float32)[None, None, :]
    return np.concatenate([c * m, m], axis=2)

def _dil(mask, r):
    r = int(round(r))
    if r <= 0: return mask
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def sticker_from_alpha(content_pm, outline=9, border=6, shadow=(7, 11), shadow_a=0.55, blur=6):
    """Wrap premultiplied float content (H,W,4) in black outline + white border + soft drop shadow (all in design px)."""
    o, b = outline * RS, border * RS
    sx, sy = shadow[0] * RS, shadow[1] * RS
    pad = int(o + b + abs(sx) + abs(sy) + blur * RS * 3 + 6)
    h, w = content_pm.shape[:2]
    big = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
    big[pad:pad + h, pad:pad + w] = content_pm
    a8 = (big[..., 3] * 255).astype(np.uint8)
    m_black = _dil(a8, o); m_white = _dil(a8, o + b)
    sh = np.roll(np.roll(m_white, int(sy), 0), int(sx), 1)
    sh = cv2.GaussianBlur(sh, (0, 0), blur * RS)
    out = _layer(sh, BLACK, shadow_a)
    out = _over(out, _layer(m_white, WHITE))
    out = _over(out, _layer(m_black, BLACK))
    out = _over(out, big)
    return out


def pm_to_u8(pm):
    a = pm[..., 3:]
    rgb = np.where(a > 1e-4, pm[..., :3] / np.maximum(a, 1e-4), 0)
    return np.concatenate([np.clip(rgb, 0, 255), a * 255], axis=2).astype(np.uint8)


# ---------------------------------------------------------------- text
def text_pm(lines, color, size, stroke=0.115, shadow=(0.06, 0.09), baloo_scale=0.93, tracking=0.0,
            align="center", leading=1.02, sizes=None):
    """Heavy outlined text with drop shadow. lines: list[str]; sizes: optional per-line design sizes.
    The U+09F3 taka glyph is not in Titan One so it is drawn from Baloo Da 2 ExtraBold, size-matched."""
    sizes = sizes or [size] * len(lines)
    runs_per_line = []; widths = []; heights = []
    for ln, sz in zip(lines, sizes):
        S = sz * RS
        ft = _font(TITAN, S); fb = _font(BALOO, S * baloo_scale, "ExtraBold")
        runs = []; x = 0
        for ch in ln:
            f = fb if ch == "৳" else ft
            adv = f.getlength(ch) + tracking * S + (0.05 * S if ch == "\u09f3" else 0)
            runs.append((ch, f, x)); x += adv
        runs_per_line.append(runs); widths.append(x); heights.append(S)
    sw = int(round(max(sizes) * RS * stroke))
    pad = sw + int(max(sizes) * RS * 0.25)
    W = int(max(widths)) + 2 * pad
    lh = [h * leading for h in heights]
    Hh = int(sum(lh) + 2 * pad + heights[0] * 0.25)
    layer_stroke = Image.new("L", (W, Hh), 0); layer_fill = Image.new("L", (W, Hh), 0)
    ds = ImageDraw.Draw(layer_stroke); df = ImageDraw.Draw(layer_fill)
    y = pad
    for runs, wd, hh, l in zip(runs_per_line, widths, heights, lh):
        base = y + hh * 0.92
        x0 = pad + ((max(widths) - wd) / 2 if align == "center" else 0)
        for ch, f, xx in runs:
            ds.text((x0 + xx, base), ch, font=f, fill=255, stroke_width=sw, stroke_fill=255, anchor="ls")
            df.text((x0 + xx, base), ch, font=f, fill=255, anchor="ls")
        y += l
    fa = np.asarray(layer_fill, np.float32) / 255; sa = np.asarray(layer_stroke, np.float32) / 255
    sa = np.maximum(sa, fa)
    stroke_pm = np.concatenate([np.zeros((Hh, W, 3), np.float32), sa[..., None]], axis=2)
    fill_pm = np.concatenate([np.array(color, np.float32)[None, None] * fa[..., None], fa[..., None]], axis=2)
    ox, oy = int(shadow[0] * max(sizes) * RS), int(shadow[1] * max(sizes) * RS)
    sh = np.roll(np.roll((sa * 255).astype(np.uint8), oy, 0), ox, 1)
    sh = cv2.GaussianBlur(sh, (0, 0), 2.5 * RS)
    out = _layer(sh, BLACK, 0.6)
    out = _over(out, stroke_pm); out = _over(out, fill_pm)
    return out


# ---------------------------------------------------------------- shapes
def burst_mask(rx, ry, points, inner, rot=0.0, wob=0.0, seed=1):
    """Comic starburst polygon mask (design px radii) as uint8 (H,W)."""
    rx *= RS; ry *= RS
    W, H = int(2 * rx + 8), int(2 * ry + 8)
    cx, cy = W / 2, H / 2
    pts = []
    rng = np.random.default_rng(seed)
    n = points * 2
    for k in range(n):
        ang = rot + math.pi * 2 * k / n
        r = 1.0 if k % 2 == 0 else inner
        if wob and k % 2 == 0: r *= 1 + rng.uniform(-wob, wob)
        pts.append((cx + math.cos(ang) * rx * r, cy + math.sin(ang) * ry * r))
    ss = 3
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.float64).reshape(-1, 1, 2).__mul__(ss).astype(np.int32)], 255)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)


def halftone(shape, color, step=20, rmax=7.5, direction=(1, 1), phase=0.0):
    """Halftone dot layer (premultiplied) with dot radius growing along `direction`."""
    H, W = shape
    st = step * RS
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # rotated 45deg grid
    u = (xx + yy) / math.sqrt(2); v = (xx - yy) / math.sqrt(2)
    gu = (u / st) - np.floor(u / st) - 0.5; gv = (v / st) - np.floor(v / st) - 0.5
    d = np.sqrt(gu ** 2 + gv ** 2) * st
    g = (xx * direction[0] + yy * direction[1]); g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    r = rmax * RS * (g ** 1.2) + phase
    a = np.clip((r - d) * 1.2 + 0.5, 0, 1)
    return np.concatenate([np.array(color, np.float32)[None, None] * a[..., None], a[..., None]], axis=2)


def burst_sprite(rx, ry, points, inner, fill, dots, rot=0.0, wob=0.0, seed=1, outline=11, border=7):
    m = burst_mask(rx, ry, points, inner, rot, wob, seed)
    base = _layer(m, fill)
    ht = halftone(m.shape, dots, step=19, rmax=6.8)
    ht[..., :] *= (m.astype(np.float32) / 255)[..., None]
    base = _over(base, ht)
    return sticker_from_alpha(base, outline, border)


def circle_badge(d, color, outline=9, border=6, dots=None):
    D = int(d * RS)
    m = np.zeros((D, D), np.uint8)
    cv2.circle(m, (D // 2, D // 2), D // 2 - 1, 255, -1, cv2.LINE_AA)
    base = _layer(m, color)
    if dots is not None:
        ht = halftone(m.shape, dots, step=16, rmax=5.5, direction=(1, 1.4))
        ht *= (m.astype(np.float32) / 255)[..., None]
        base = _over(base, ht)
    return sticker_from_alpha(base, outline, border)


# ---------------------------------------------------------------- icons (drawn as straight RGBA with PIL @ 3x supersample)
def _icon_canvas(S=300, ss=3):
    im = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0)); return im, ImageDraw.Draw(im), ss

def _icon_done(im, S, ss, out_px):
    im = im.resize((int(out_px * RS), int(out_px * RS)), Image.LANCZOS)
    a = np.asarray(im, np.uint8)
    return a

def icon_bag(px):
    im, d, s = _icon_canvas()
    k = lambda v: v * s
    d.arc((k(95), k(40), k(205), k(150)), 180, 360, fill=BLACK, width=k(15))      # handle
    d.rounded_rectangle((k(55), k(100), k(245), k(262)), radius=k(24), fill=WHITE, outline=BLACK, width=k(13))
    d.rounded_rectangle((k(66), k(112), k(234), k(150)), radius=k(14), fill=CYAN)  # stripe
    d.ellipse((k(110), k(163), k(120), k(173)), fill=BLACK); d.ellipse((k(180), k(163), k(190), k(173)), fill=BLACK)
    d.arc((k(125), k(172), k(175), k(215)), 20, 160, fill=BLACK, width=k(10))      # smile
    return _icon_done(im, 300, s, px)

def icon_phone(px):
    im, d, s = _icon_canvas()
    k = lambda v: v * s
    d.rounded_rectangle((k(88), k(30), k(212), k(270)), radius=k(30), fill=WHITE, outline=BLACK, width=k(13))
    d.rounded_rectangle((k(104), k(62), k(196), k(232)), radius=k(12), fill=CYAN, outline=BLACK, width=k(7))
    d.ellipse((k(118), k(112), k(182), k(176)), fill=BLACK)                         # lens
    d.ellipse((k(133), k(127), k(167), k(161)), fill=PURPLE)
    d.ellipse((k(144), k(135), k(156), k(147)), fill=WHITE)
    d.ellipse((k(165), k(72), k(187), k(94)), fill=PINK, outline=BLACK, width=k(4))  # rec dot
    d.rounded_rectangle((k(132), k(244), k(168), k(254)), radius=k(5), fill=BLACK)
    return _icon_done(im, 300, s, px)

def icon_receipt(px):
    im, d, s = _icon_canvas()
    k = lambda v: v * s
    pts = [(78, 36), (222, 36), (222, 262)]
    n = 6; x0, x1 = 222, 78
    for i in range(n):
        xa = x0 + (x1 - x0) * (i + 0.5) / n; xb = x0 + (x1 - x0) * (i + 1) / n
        pts += [(xa, 238), (xb, 262)]
    pts += [(78, 262)]
    P = [(k(x), k(y)) for x, y in pts]
    d.polygon(P, fill=WHITE)
    d.line(P + [P[0]], fill=BLACK, width=k(13), joint="curve")
    for y, wdt in ((82, 110), (116, 80), (150, 100)):
        d.rounded_rectangle((k(105), k(y), k(105 + wdt), k(y + 14)), radius=k(7), fill=BLACK)
    d.ellipse((k(112), k(182), k(176), k(232)), fill=PINK, outline=BLACK, width=k(7))
    d.line([(k(128), k(207)), (k(141), k(220)), (k(162), k(192))], fill=WHITE, width=k(10), joint="curve")
    return _icon_done(im, 300, s, px)

def icon_tag(px):
    im, d, s = _icon_canvas()
    k = lambda v: v * s
    P = [(34, 150), (104, 62), (258, 62), (258, 238), (104, 238)]
    P = [(k(x), k(y)) for x, y in P]
    d.polygon(P, fill=CYAN); d.line(P + [P[0]], fill=BLACK, width=k(14), joint="curve")
    d.ellipse((k(86), k(130), k(122), k(166)), fill=WHITE, outline=BLACK, width=k(8))
    f = _font(TITAN, 120 * s)
    d.text((k(190), k(152)), "%", font=f, fill=PINK, anchor="mm", stroke_width=k(7), stroke_fill=BLACK)
    return _icon_done(im, 300, s, px)


def badge_with_icon(color, icon_fn, d=170, icon_frac=1.0, rot=0.0, dots=None):
    badge = circle_badge(d, color, dots=dots)
    ic = icon_fn(int(d * icon_frac))
    icm = np.concatenate([ic[..., :3].astype(np.float32) * (ic[..., 3:] / 255), ic[..., 3:] / 255], axis=2)
    if rot:
        h, w = icm.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
        icm = cv2.warpAffine(icm, M, (w, h))
    H, W = badge.shape[:2]; h, w = icm.shape[:2]
    y0 = (H - h) // 2; x0 = (W - w) // 2
    ov = np.zeros_like(badge); ov[y0:y0 + h, x0:x0 + w] = icm
    return _over(badge, ov)


def hcompose(parts, gap=0, valign="center"):
    """Lay premultiplied sprites left->right on one canvas."""
    H = max(p.shape[0] for p in parts)
    W = sum(p.shape[1] for p in parts) + int(gap * RS) * (len(parts) - 1)
    out = np.zeros((H, W, 4), np.float32); x = 0
    for p in parts:
        y = (H - p.shape[0]) // 2
        out[y:y + p.shape[0], x:x + p.shape[1]] = _over(out[y:y + p.shape[0], x:x + p.shape[1]], p)
        x += p.shape[1] + int(gap * RS)
    return out


def to_sprite(pm):
    """premultiplied float (RS space) -> stored sprite at SSF."""
    return finalize(pm_to_u8(pm))
