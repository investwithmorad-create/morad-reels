"""Morad 3D map engine (Morad Pulse 2.0, Map gear).

Renders a 9:16 fly-in over real satellite imagery (Esri World Imagery tiles):
camera push-in and tilt, plot boundary drawing itself with gold glow, area dimming,
an illustrative tower massing rising with shadow, route lines and pins appearing one
at a time, labels in the brand fonts. Output: PNG frames -> MP4 via ffmpeg.

Usage:  python3 map3d.py scene.json out.mp4 [--draft]
"""
import json, math, os, sys, subprocess, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import tiles

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = {
    'bebas': os.path.join(HERE, 'fonts/BebasNeue-Regular.ttf'),
    'mont': os.path.join(HERE, 'fonts/Montserrat-VF.ttf'),
    'inter': os.path.join(HERE, 'fonts/Inter-VF.ttf'),
}
GOLD = np.array([98, 184, 222], np.float32)      # BGR of (222,184,98)
GOLD_HI = np.array([160, 228, 250], np.float32)
GOLD_LO = np.array([42, 122, 168], np.float32)
EARTH = 40075016.686


def font(name, size, weight=None):
    f = ImageFont.truetype(FONT[name], size)
    if weight:
        try:
            f.set_variation_by_axes([weight] if name == 'mont' else [14, weight])
        except Exception:
            pass
    return f


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


class World:
    """Local ground metres around an origin (Web Mercator scaled by cos(lat0))."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.k = math.cos(math.radians(lat0))
        self.ox, self.oy = tiles.lonlat_to_px(lon0, lat0, 0)

    def xy(self, lon, lat):
        x, y = tiles.lonlat_to_px(lon, lat, 0)
        s = EARTH * self.k / 256
        return np.array([(x - self.ox) * s, -(y - self.oy) * s])


class Layer:
    """One mosaic with a mip pyramid, georeferenced in world metres."""

    def __init__(self, world, bbox, z):
        lon0, lat0, lon1, lat1 = bbox
        img, px0, py0 = tiles.mosaic(lon0, lat0, lon1, lat1, z)
        self.mpp = EARTH * world.k / (256 * 2 ** z)
        s0 = EARTH * world.k / 256
        # world X of mosaic pixel u: (px0+u)/2^z*s0 - ox*s0
        self.x0 = (px0 / 2 ** z - world.ox) * s0
        self.y0 = -(py0 / 2 ** z - world.oy) * s0
        self.levels = [img.astype(np.uint8)]
        while min(self.levels[-1].shape[:2]) > 64 and len(self.levels) < 9:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        self.h, self.w = img.shape[:2]


class Camera:
    def __init__(self, W, H, vfov=46):
        self.W, self.H = W, H
        self.f = (H / 2) / math.tan(math.radians(vfov / 2))

    def set(self, target, dist, heading, pitch):
        h, p = math.radians(heading), math.radians(pitch)
        f = np.array([math.sin(p) * math.sin(h), math.sin(p) * math.cos(h), -math.cos(p)])
        r = np.array([math.cos(h), -math.sin(h), 0.0])
        u = np.cross(r, f)
        self.R = np.stack([r, -u, f])  # rows: x right, y down, z forward
        self.C = np.array([target[0], target[1], 0.0]) - dist * f
        return self

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, np.float64))
        if P.shape[1] == 2:
            P = np.hstack([P, np.zeros((len(P), 1))])
        c = (P - self.C) @ self.R.T
        z = np.maximum(c[:, 2], 1e-3)
        return np.stack([self.f * c[:, 0] / z + self.W / 2, self.f * c[:, 1] / z + self.H / 2], 1), c[:, 2]


def render_ground(cam, layers, haze=(36, 30, 26)):
    W, H = cam.W, cam.H
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
    d = np.stack([(xs - W / 2) / cam.f, (ys - H / 2) / cam.f, np.ones_like(xs)], -1) @ cam.R  # world dirs
    dz = d[..., 2]
    sky = dz > -1e-4
    t = np.where(sky, 1e9, -cam.C[2] / np.minimum(dz, -1e-4))
    X = cam.C[0] + t * d[..., 0]
    Y = cam.C[1] + t * d[..., 1]
    dn = np.linalg.norm(d, axis=-1)
    foot = (t * dn / cam.f) / np.maximum(np.abs(dz) / dn, 0.15)  # metres per output pixel
    out = np.zeros((H, W, 3), np.float32)
    filled = np.zeros((H, W), bool)
    for L in layers:  # finest first
        U = (X - L.x0) / L.mpp
        V = (L.y0 - Y) / L.mpp
        inside = (~sky) & (~filled) & (U >= 1) & (V >= 1) & (U < L.w - 2) & (V < L.h - 2)
        if not inside.any():
            continue
        lvl = np.clip(np.floor(np.log2(np.maximum(foot / L.mpp, 1e-6)) + 0.5), 0, len(L.levels) - 1).astype(int)
        for k in np.unique(lvl[inside]):
            m = inside & (lvl == k)
            sc = 2 ** k
            src = L.levels[k]
            samp = cv2.remap(src, (U / sc).astype(np.float32), (V / sc).astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out[m] = samp[m][:, ::-1]  # RGB -> BGR
        filled |= inside
    # fill gaps (beyond imagery) and sky with haze; distance haze
    hz = np.array(haze, np.float32)
    dist = np.where(sky, 1e9, t * dn)
    cd = float(np.linalg.norm(cam.C))  # camera distance scale
    a = np.clip((dist - max(9000, 1.4 * cam.C[2] / max(math.cos(0.0), 1e-3))) / max(30000, 2.5 * cam.C[2]), 0, 0.75)[..., None]
    out = np.where(filled[..., None], out * (1 - a) + hz * a, hz)
    # sky gradient above horizon
    if sky.any():
        g = np.clip((ys / H), 0, 1)[..., None]
        skycol = np.array([22, 16, 12], np.float32) * (1 - g) + hz * g
        out = np.where(sky[..., None], skycol, out)
    return out, filled


def poly_mask(pts, W, H):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255


def glow_line(img, pts, width, color, glow=18, alpha=1.0, closed=False):
    if len(pts) < 2:
        return
    H, W = img.shape[:2]
    lay = np.zeros((H, W), np.float32)
    cv2.polylines(lay, [np.round(pts * 4).astype(np.int32)], closed, 1.0, max(1, int(width)), cv2.LINE_AA, shift=2)
    g = cv2.GaussianBlur(lay, (0, 0), glow)
    core = lay * alpha
    gl = np.clip(g * 2.2, 0, 1) * 0.55 * alpha
    img[:] = img * (1 - core[..., None]) + GOLD_HI * core[..., None]
    img[:] = img + GOLD[None, None, :] * gl[..., None]


def partial(poly, frac, closed=True):
    pts = list(poly) + ([poly[0]] if closed else [])
    seg = [np.linalg.norm(np.subtract(pts[i + 1], pts[i])) for i in range(len(pts) - 1)]
    total = sum(seg) * frac
    out = [pts[0]]
    for i, s in enumerate(seg):
        if total >= s:
            out.append(pts[i + 1]); total -= s
        else:
            if s > 0:
                out.append(np.add(pts[i], np.subtract(pts[i + 1], pts[i]) * (total / s)))
            break
    return np.array(out)


def draw_tower(img, cam, foot, height, sun=(230, 40)):
    """Illustrative massing: dark glass walls, faint floor lines, gold crown, soft shadow."""
    if height < 1:
        return
    H, W = img.shape[:2]
    foot = np.asarray(foot, float)
    az, el = math.radians(sun[0]), math.radians(sun[1])
    L = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    off = -L[:2] / L[2] * height
    shadow = np.vstack([foot, foot + off])
    hull = cv2.convexHull(cam.project(shadow)[0].astype(np.float32))[:, 0, :]
    sm = cv2.GaussianBlur(poly_mask(hull, W, H), (0, 0), 3)
    img[:] = img * (1 - 0.5 * sm[..., None])
    n = len(foot)
    top = np.hstack([foot, np.full((n, 1), height)])
    base = np.hstack([foot, np.zeros((n, 1))])
    cen = np.append(foot.mean(0), height / 2)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([base[i], base[j], top[j], top[i]])
        mid = quad.mean(0)
        e = foot[j] - foot[i]
        nrm = np.array([e[1], -e[0], 0.0]); nrm /= np.linalg.norm(nrm) + 1e-9
        if np.dot(nrm, mid - cen) < 0:
            nrm = -nrm
        if np.dot(nrm, cam.C - mid) <= 0:
            continue
        faces.append((np.linalg.norm(cam.C - mid), quad, nrm, i))
    faces.sort(key=lambda a: -a[0])
    for _, quad, nrm, i in faces:
        p2, _ = cam.project(quad)
        lit = 0.35 + 0.65 * max(np.dot(nrm, L), 0)
        m = poly_mask(p2, W, H)[..., None]
        # vertical gradient glass: darker at base, reflective near top
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        lo, hi = p2[:, 1].min(), p2[:, 1].max()
        g = np.clip((np.arange(H)[:, None, None] - lo) / max(hi - lo, 1), 0, 1).astype(np.float32)
        base_c = np.array([46, 36, 28], np.float32) * (0.7 + 0.6 * lit)
        top_c = np.array([150, 130, 112], np.float32) * (0.45 + 0.55 * lit)
        col = top_c * (1 - g) + base_c * g
        img[:] = img * (1 - m * 0.96) + col * m * 0.96
        # floor lines every ~4 m (thin, faint)
        step = 4.0
        for k in range(1, int(height / step)):
            hgt = k * step
            a3 = quad[0].copy(); a3[2] = hgt
            b3 = quad[1].copy(); b3[2] = hgt
            ab, _ = cam.project(np.array([a3, b3]))
            cv2.line(img, tuple(np.round(ab[0] * 4).astype(int)), tuple(np.round(ab[1] * 4).astype(int)),
                     (90, 82, 74), 1, cv2.LINE_AA, shift=2)
        # corner edges in soft gold
        edge, _ = cam.project(np.array([quad[0], quad[3]]))
        cv2.line(img, tuple(np.round(edge[0] * 4).astype(int)), tuple(np.round(edge[1] * 4).astype(int)),
                 tuple(float(c) for c in GOLD * 0.8), 1, cv2.LINE_AA, shift=2)
    tp, _ = cam.project(top)
    m = poly_mask(tp, W, H)[..., None]
    img[:] = img * (1 - m) + np.array([120, 112, 104], np.float32) * m
    glow_line(img, tp, 2, GOLD, glow=6, alpha=0.95, closed=True)


def text_img(img, x, y, txt, fnt, fill=(255, 255, 255), anchor='la', alpha=1.0, shadow=True):
    pil = Image.fromarray(np.clip(img[..., ::-1], 0, 255).astype(np.uint8))
    lay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if shadow:
        d.text((x + 2, y + 3), txt, font=fnt, fill=(0, 0, 0, int(150 * alpha)), anchor=anchor)
    d.text((x, y), txt, font=fnt, fill=tuple(fill) + (int(255 * alpha),), anchor=anchor)
    pil = Image.alpha_composite(pil.convert('RGBA'), lay).convert('RGB')
    img[:] = np.asarray(pil, np.float32)[..., ::-1]


def pill(img, x, y, txt, fnt, alpha=1.0, scale=1.0):
    """Dark glass label with a gold bar on the left, anchored bottom-centre at (x, y)."""
    if alpha <= 0:
        return
    pil = Image.fromarray(np.clip(img[..., ::-1], 0, 255).astype(np.uint8)).convert('RGBA')
    lay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    tw = d.textlength(txt, font=fnt)
    w, h = tw + 44 * scale, 58 * scale
    x0, y0 = x - w / 2, y - h
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10 * scale, fill=(12, 12, 14, int(215 * alpha)),
                        outline=(222, 184, 98, int(200 * alpha)), width=2)
    d.rectangle([x0 + 10 * scale, y0 + 14 * scale, x0 + 15 * scale, y0 + h - 14 * scale], fill=(222, 184, 98, int(255 * alpha)))
    if alpha > 0.3:
        d.text((x0 + 26 * scale, y0 + h / 2), txt, font=fnt, fill=(255, 255, 255, int(255 * alpha)), anchor='lm')
    pil = Image.alpha_composite(pil, lay).convert('RGB')
    img[:] = np.asarray(pil, np.float32)[..., ::-1]


def pin(img, p, alpha=1.0, r=11):
    if alpha <= 0:
        return
    H, W = img.shape[:2]
    lay = np.zeros((H, W), np.float32)
    cv2.circle(lay, (int(p[0]), int(p[1])), int(r * 2.2), 1.0, -1, cv2.LINE_AA)
    g = cv2.GaussianBlur(lay, (0, 0), 10)
    img[:] = img + GOLD[None, None, :] * (g * 0.6 * alpha)[..., None]
    cv2.circle(img, (int(p[0]), int(p[1])), r, tuple(float(c) for c in GOLD_HI), -1, cv2.LINE_AA)
    cv2.circle(img, (int(p[0]), int(p[1])), int(r * 0.45), (14, 12, 12), -1, cv2.LINE_AA)


def interp_cam(keys, t, world):
    if t <= keys[0]['t']:
        a = b = keys[0]; s = 0
    elif t >= keys[-1]['t']:
        a = b = keys[-1]; s = 0
    else:
        for i in range(len(keys) - 1):
            if keys[i]['t'] <= t <= keys[i + 1]['t']:
                a, b = keys[i], keys[i + 1]
                s = ease((t - a['t']) / max(b['t'] - a['t'], 1e-6))
                break
    ta, tb = world.xy(a['lon'], a['lat']), world.xy(b['lon'], b['lat'])
    tgt = ta + (tb - ta) * s
    dist = math.exp(math.log(a['dist']) * (1 - s) + math.log(b['dist']) * s)
    hd = a['heading'] + (b['heading'] - a['heading']) * s
    pt = a['pitch'] + (b['pitch'] - a['pitch']) * s
    return tgt, dist, hd, pt


def main(scene_path, out_path, draft=False, stills=None):
    S = json.load(open(scene_path))
    W, H = (540, 960) if draft else (1080, 1920)
    sc = W / 1080
    fps = S.get('fps', 25)
    world = World(S['origin'][0], S['origin'][1])
    layers = [Layer(world, L['bbox'], L['z']) for L in S['layers']]  # finest first
    cam = Camera(W, H, S.get('vfov', 46))
    plot = np.array([world.xy(*p) for p in S['plot']['polygon']])
    tower = S.get('tower')
    tfoot = np.array([world.xy(*p) for p in tower['footprint']]) if tower else None
    pins = [dict(p, xy=world.xy(p['lon'], p['lat'])) for p in S.get('pins', [])]
    plot_c = plot.mean(0)
    F = {
        'title': font('bebas', int(92 * sc)), 'sub': font('mont', int(32 * sc), 700),
        'pin': font('mont', int(30 * sc), 700), 'note': font('inter', int(22 * sc), 500),
        'area': font('bebas', int(64 * sc)),
    }
    tmp = out_path + '_frames'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    N = int(S['duration'] * fps)
    ev = S['events']
    frames = range(N) if not stills else [int(round(x * fps)) for x in stills]
    for fi in frames:
        t = fi / fps
        tgt, dist, hd, pt = interp_cam(S['camera'], t, world)
        cam.set(tgt, dist, hd, pt)
        img, filled = render_ground(cam, layers)
        # area highlight: dim and desaturate outside the plot
        pp, _ = cam.project(plot)
        hl = 0
        if 'highlight' in ev:
            h0, h1 = ev['highlight'][:2]
            hl = ease((t - h0) / (h1 - h0))
            if len(ev['highlight']) == 4:
                hl *= 1 - 0.8 * ease((t - ev['highlight'][2]) / (ev['highlight'][3] - ev['highlight'][2]))
        if hl > 0:
            m = cv2.GaussianBlur(poly_mask(pp, W, H), (0, 0), 2 * sc + 0.5)[..., None]
            grey = img.mean(-1, keepdims=True)
            outside = (img * 0.35 + grey * 0.65) * 0.5
            img = img * m + (img * (1 - hl) + outside * hl) * (1 - m)
            img = img + GOLD * 0.10 * hl * m
        # boundary drawing itself
        if 'boundary' in ev:
            b0, b1 = ev['boundary']
            fr = ease_out((t - b0) / (b1 - b0))
            if fr > 0:
                glow_line(img, partial(pp, fr), max(2, 4 * sc), GOLD, glow=14 * sc, alpha=1.0, closed=False)
        # routes and pins, one at a time
        for k, p in enumerate(pins):
            r0 = p['t']
            fr = ease_out((t - r0) / 1.0)
            if fr <= 0:
                continue
            span = np.linalg.norm(p['xy'] - plot_c)
            line3 = np.array([np.append(plot_c + (p['xy'] - plot_c) * s, 4 * 0.10 * span * s * (1 - s)) for s in np.linspace(0, fr, 60)])
            l2, _ = cam.project(line3)
            glow_line(img, l2, max(2, 3 * sc), GOLD, glow=10 * sc, alpha=0.95)
            if fr >= 1:
                a = ease((t - r0 - 1.0) / 0.35)
                ep, z = cam.project(p['xy'])
                if z[0] > 0:
                    pin(img, ep[0], a, r=int(11 * sc))
                    pill(img, ep[0][0], ep[0][1] - 22 * sc, p['name'], F['pin'], alpha=a, scale=sc * (0.85 + 0.15 * a))
        # tower rising (illustrative massing)
        if tower:
            r0, r1 = ev['tower']
            hgt = tower['height'] * ease_out((t - r0) / (r1 - r0))
            draw_tower(img, cam, tfoot, hgt)
        # plot pin label
        if 'plot_label' in ev and t >= ev['plot_label']:
            a = ease((t - ev['plot_label']) / 0.4)
            top = np.append(plot_c, (tower['height'] if tower else 0) * 1.05)
            if not ('pins_hide_plot' in ev and t >= ev['pins_hide_plot']):
                lp, _ = cam.project(top)
                pill(img, lp[0][0], lp[0][1] - 16 * sc, S['plot']['label'], F['pin'], alpha=a, scale=sc)
        # Pulse label (title + gold rule + sub-line), y ~860
        if 'label' in S and 'label' in ev:
            l0, l1 = ev['label']
            a = ease((t - l0) / 0.4) * (1 - ease((t - l1) / 0.4))
            if a > 0:
                x = 135 * sc
                ly = S['label'].get('y', 860)
                rule = 220 * sc * ease_out((t - l0) / 0.6)
                cv2.line(img, (int(x), int((ly - 58) * sc)), (int(x + rule), int((ly - 58) * sc)), tuple(float(c) for c in GOLD), max(1, int(4 * sc)), cv2.LINE_AA)
                pil = Image.fromarray(np.clip(img[..., ::-1], 0, 255).astype(np.uint8)).convert('RGBA')
                lay = Image.new('RGBA', pil.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
                ttl, sub = S['label']['title'], S['label']['sub']
                bb = d.textbbox((x, ly * sc), ttl, font=F['title'], anchor='lm')
                d.rectangle([bb[0] - 18 * sc, bb[1] - 14 * sc, bb[2] + 18 * sc, bb[3] + 14 * sc], fill=(0, 0, 0, int(225 * a)))
                d.text((x, ly * sc), ttl, font=F['title'], fill=(255, 255, 255, int(255 * a)), anchor='lm')
                bb2 = d.textbbox((x, (ly + 88) * sc), sub, font=F['sub'], anchor='lm')
                d.rectangle([bb2[0] - 14 * sc, bb2[1] - 10 * sc, bb2[2] + 14 * sc, bb2[3] + 10 * sc], fill=(12, 12, 14, int(215 * a)))
                d.text((x, (ly + 88) * sc), sub, font=F['sub'], fill=(222, 184, 98, int(255 * a)), anchor='lm')
                pil = Image.alpha_composite(pil, lay).convert('RGB')
                img = np.asarray(pil, np.float32)[..., ::-1].copy()
        # credit + note (inside safe zone)
        text_img(img, 65 * sc, 1236 * sc, S.get('credit', 'Imagery: Esri, Maxar'), F['note'], fill=(220, 220, 220), anchor='ls', alpha=0.85)
        if S.get('note'):
            text_img(img, 888 * sc, 1236 * sc, S['note'], F['note'], fill=(222, 184, 98), anchor='rs', alpha=0.9)
        # light vignette
        if fi == frames[0]:
            yy, xx = np.mgrid[0:H, 0:W]
            vig = 1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
            vig = np.clip(vig, 0.6, 1)[..., None].astype(np.float32)
        img = img * vig
        cv2.imwrite(os.path.join(tmp, f'{fi:05d}.png'), np.clip(img, 0, 255).astype(np.uint8))
        if fi % 25 == 0:
            print(f'frame {fi}/{N}', flush=True)
    if stills:
        print('stills in', tmp); return
    crf = '28' if draft else '14'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', os.path.join(tmp, '%05d.png'),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', crf, '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', out_path], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print('done', out_path)


if __name__ == '__main__':
    st = [a for a in sys.argv if a.startswith('--stills=')]
    main(sys.argv[1], sys.argv[2], '--draft' in sys.argv, [float(x) for x in st[0][9:].split(',')] if st else None)
