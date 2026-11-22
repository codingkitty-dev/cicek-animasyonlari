import math
import numpy as np
import pygame

# ── Pencere ve Temel Ayarlar ─────────────────────────────────────────────────
W, H = 760, 860
FPS = 40
GX, GY = W * 0.5, H * 0.87   # Toprak tepeciği merkezi
STEM_H = 410.0                # Sap yüksekliği

rng = np.random.default_rng(42)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def midrib(L, th0, curl, n=40):
    us = np.linspace(0.0, 1.0, n + 1)
    a = th0 + curl * (us ** 1.6)
    ds = L / n
    r = np.concatenate([[0.0], np.cumsum(np.sin(a[:-1]) * ds)])
    h = np.concatenate([[0.0], np.cumsum(np.cos(a[:-1]) * ds)])
    return us, r, h


# ── Arka Plan Gradyanı ve Yıldızlar ──────────────────────────────────────────
yy = np.linspace(0.0, 1.0, H)[:, None]
bg_base = np.zeros((H, W, 3), dtype=np.float32)
bg_base[:, :, 0] = lerp(4.0, 14.0, yy ** 1.4)
bg_base[:, :, 1] = lerp(7.0, 19.0, yy ** 1.3)
bg_base[:, :, 2] = lerp(18.0, 42.0, yy ** 1.1)

# Yıldızlar
N_STARS = 260
star_x = rng.integers(0, W, N_STARS)
star_y = rng.integers(0, int(H * 0.82), N_STARS)
star_ph = rng.uniform(0.0, math.tau, N_STARS)
star_sp = rng.uniform(1.2, 3.2, N_STARS)
star_br = rng.uniform(110.0, 245.0, N_STARS)
star_col = np.stack([
    star_br * rng.uniform(0.88, 1.0, N_STARS),
    star_br * rng.uniform(0.92, 1.0, N_STARS),
    star_br,
], axis=1)

# Toprak Tepeciği (Işıltılı Noktalı Zemin)
N_SOIL = 6800
so_dx = rng.normal(0.0, W * 0.27, N_SOIL)
so_dx = np.clip(so_dx, -W * 0.5, W * 0.5)
hill_top = GY - 42.0 * np.exp(-(so_dx / (W * 0.25)) ** 2)
so_depth = rng.uniform(0.0, 1.0, N_SOIL) ** 0.72
so_y = hill_top + so_depth * (H - hill_top + 12.0)
ridge = np.exp(-so_depth * 4.5) * np.exp(-(so_dx / (W * 0.34)) ** 2)
so_fade = (1.0 - so_depth * 0.82) * np.exp(-(so_dx / (W * 0.40)) ** 2)
so_col = np.stack([
    lerp(35.0, 185.0, np.clip(so_fade * 0.65 + ridge * 0.55, 0.0, 1.0)),
    lerp(42.0, 192.0, np.clip(so_fade * 0.65 + ridge * 0.55, 0.0, 1.0)),
    lerp(62.0, 220.0, np.clip(so_fade * 0.65 + ridge * 0.55, 0.0, 1.0)),
], axis=1) * rng.uniform(0.70, 1.15, (N_SOIL, 1))

# ── Sap (Gövde) ve Dikenler ──────────────────────────────────────────────────
N_STEM = 3600
st_s = rng.uniform(0.0, 1.0, N_STEM)
st_w = lerp(7.2, 3.8, st_s)
st_dx = rng.normal(0.0, 0.42, N_STEM) * st_w
st_edge = np.clip(np.abs(st_dx) / (st_w * 0.85), 0.0, 1.0)
st_col = np.stack([
    lerp(42.0, 145.0, st_edge),
    lerp(135.0, 245.0, st_edge),
    lerp(68.0, 145.0, st_edge),
], axis=1) * rng.uniform(0.82, 1.12, (N_STEM, 1))

# Gül Dikenleri
thorns = []
for s_pos, side in [(0.20, -1), (0.39, 1), (0.58, -1), (0.74, 1)]:
    nt = 100
    tu = rng.uniform(0.0, 1.0, nt)
    tv = rng.uniform(-1.0, 1.0, nt) * (1.0 - tu)
    thorns.append(dict(
        s=s_pos,
        side=side,
        u=tu,
        v=tv,
        col=np.stack([
            lerp(68.0, 170.0, tu),
            lerp(140.0, 210.0, tu),
            lerp(80.0, 135.0, tu),
        ], axis=1),
    ))

# ── Gül Yaprakları ───────────────────────────────────────────────────────────
leaves = []
leaf_specs = [
    (0.22, 145.0,  1, 0.96, 0.62),
    (0.28, 158.0, -1, 0.98, 0.65),
    (0.45, 138.0,  1, 0.86, 0.56),
    (0.51, 144.0, -1, 0.88, 0.58),
    (0.66, 108.0,  1, 0.78, 0.48),
]
for s_pos, lf_len, side, th0, curl in leaf_specs:
    nl = 1750
    u = rng.uniform(0.0, 1.0, nl) ** 0.86
    raw_v = rng.uniform(-1.0, 1.0, nl)
    v = np.sign(raw_v) * (np.abs(raw_v) ** 0.70)
    # Zarif, hafif tırtıklı gül yaprağı formu
    serr = 1.0 + 0.06 * np.sin(u * 26.0) * np.sin(u * np.pi)
    width = 27.0 * (np.sin(u * np.pi) ** 0.76) * (1.0 - 0.16 * u) * serr
    edge = np.clip(np.abs(v) ** 1.35, 0.0, 1.0)
    vein = np.exp(-((v * 5.2) ** 2)) * 0.42
    col = np.stack([
        lerp(40.0, 165.0, np.clip(edge + vein, 0.0, 1.0)),
        lerp(142.0, 252.0, np.clip(edge * 0.85 + vein + 0.25 * u, 0.0, 1.0)),
        lerp(72.0, 158.0, np.clip(edge + 0.22 * u, 0.0, 1.0)),
    ], axis=1) * rng.uniform(0.84, 1.12, (nl, 1))
    leaves.append(dict(
        s=s_pos, L=lf_len, side=side, th0=th0, curl=curl,
        u=u, v=v, w=width, col=col,
    ))

# ── Çanak Yapraklar (Sepals - Gül Tomurcuğu Altı) ────────────────────────────
sepals = []
for k in range(5):
    ns = 480
    phi = k * (math.tau / 5.0) + 0.25
    u = rng.uniform(0.0, 1.0, ns) ** 0.85
    v = rng.uniform(-1.0, 1.0, ns)
    w = 14.0 * ((1.0 - u) ** 0.72) * (0.35 + 0.65 * np.sin(u * np.pi * 0.85))
    edge = np.abs(v)
    col = np.stack([
        lerp(45.0, 145.0, edge),
        lerp(140.0, 238.0, np.clip(edge * 0.7 + 0.3 * u, 0.0, 1.0)),
        lerp(68.0, 138.0, edge),
    ], axis=1) * rng.uniform(0.85, 1.12, (ns, 1))
    sepals.append(dict(phi=phi, u=u, v=v, w=w, col=col))

# ── Beyaz Gül Taç Yaprakları (26 Katmanlı Fibonacci Spiral) ──────────────────
N_PETALS = 26
GOLDEN_ANGLE = 2.399963229728653  # ~137.508 derece

petals = []
for k in range(N_PETALS):
    tier = k / (N_PETALS - 1)  # 0.0 = en iç kalp, 1.0 = en dış yaprak
    phi = k * GOLDEN_ANGLE + float(rng.uniform(-0.07, 0.07))

    # Dış yapraklar büyük olduğu için daha fazla nokta -> dolgun kadife doku
    np_k = int(lerp(1250, 2450, tier))
    u = rng.uniform(0.0, 1.0, np_k) ** 0.74
    rv = rng.uniform(-1.0, 1.0, np_k)
    v = np.sign(rv) * (np.abs(rv) ** 0.72)

    # Gül yaprağı genişlik profili (dolgun yuvarlak üst kenar)
    prof = (np.sin(u * math.pi) ** 0.50) * (0.40 + 0.60 * np.sin(u * math.pi * 0.76))
    tip_notch = 1.0 + 0.09 * np.exp(-((v * 2.6) ** 2)) * (u ** 2.8)

    # Klasik yüksek kadeh beyaz gül oranları
    L = lerp(56.0, 134.0, tier ** 0.68)
    W_max = lerp(40.0, 114.0, tier ** 0.65)

    # Açılma açıları: iç katmanlar dik ve sıkı spiral, dış katmanlar zarifçe açılır
    th_closed = lerp(0.07, 0.27, tier)
    th_open = lerp(0.14, 1.04, tier ** 0.95)
    curl_closed = lerp(-0.34, -0.18, tier)
    curl_open = lerp(-0.24, 0.56, tier ** 1.15)

    # Açılma zamanlaması: dıştan içe doğru açılır
    delay = lerp(0.36, 0.0, tier ** 0.85)

    # Beyaz Gül Renk Paleti (İnci Beyazı + İç Sıcak Fildişi + Ay Işığı Derinliği)
    rim = np.clip((u ** 2.1) * 0.76 + (np.abs(v) ** 1.9) * 0.58, 0.0, 1.0)
    inner_warmth = max(0.0, 1.0 - tier * 2.2)

    base_r = lerp(128.0, 222.0, inner_warmth)
    base_g = lerp(140.0, 212.0, inner_warmth)
    base_b = lerp(172.0, 188.0, inner_warmth)

    mid_r = lerp(228.0, 248.0, inner_warmth * 0.65)
    mid_g = lerp(234.0, 244.0, inner_warmth * 0.65)
    mid_b = lerp(248.0, 232.0, inner_warmth * 0.65)

    t_body = smooth(u * 1.45)
    cr = lerp(base_r, mid_r, t_body)
    cg = lerp(base_g, mid_g, t_body)
    cb = lerp(base_b, mid_b, t_body)

    # Kenarlarda ve kıvrık uçta saf parlak ay ışığı beyazı
    cr = lerp(cr, 255.0, rim * 0.90)
    cg = lerp(cg, 255.0, rim * 0.90)
    cb = lerp(cb, 255.0, rim * 0.88)

    jitter = rng.uniform(0.88, 1.08, np_k)
    col = np.clip(np.stack([cr * jitter, cg * jitter, cb * jitter], axis=1), 0.0, 255.0)
    ripple_phase = float(rng.uniform(0.0, math.tau))

    petals.append(dict(
        tier=tier, phi=phi, u=u, v=v, prof=prof, tip_notch=tip_notch,
        L=L, W_max=W_max, th_closed=th_closed, th_open=th_open,
        curl_closed=curl_closed, curl_open=curl_open, delay=delay,
        col=col, ripple_phase=ripple_phase,
    ))

# ── Uçuşan Parıltı / Ay Işığı Zerreleri ──────────────────────────────────────
N_MOTES = 75
mote_a = rng.uniform(0.0, math.tau, N_MOTES)
mote_r = rng.uniform(35.0, 225.0, N_MOTES)
mote_sp = rng.uniform(12.0, 30.0, N_MOTES)
mote_ph = rng.uniform(0.0, 10.0, N_MOTES)

# Geometrik Halka Noktaları
RING_PTS = np.linspace(0.0, math.tau, 300, endpoint=False)
OUTER_RING_PTS = np.linspace(0.0, math.tau, 420, endpoint=False)

# 3D Kamera Eğimi (~24 derece öne eğik -> gülün spiral kalbi görünür)
PITCH = 0.42
COS_P = math.cos(PITCH)
SIN_P = math.sin(PITCH)


def render(t, buf, zbuf):
    buf[:] = bg_base
    zbuf.fill(-1e9)

    def add(xs, ys, z, col, alpha=1.0):
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not np.any(m):
            return
        xi = xi[m]
        yi = yi[m]
        if np.ndim(z) == 0:
            zi = np.full(xi.shape, z, dtype=np.float32)
        else:
            zi = z[m].astype(np.float32)
        ci = col[m] if col.ndim == 2 else col

        ok = zi >= zbuf[yi, xi]
        if not np.any(ok):
            return
        xi = xi[ok]
        yi = yi[ok]
        zbuf[yi, xi] = zi[ok]
        c_sel = ci[ok] if np.ndim(ci) == 2 else ci
        if alpha >= 0.999:
            buf[yi, xi] = c_sel
        else:
            buf[yi, xi] = buf[yi, xi] * (1.0 - alpha) + c_sel * alpha

    def add_glow(xs, ys, col, alpha=0.5):
        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not np.any(m):
            return
        xi = xi[m]
        yi = yi[m]
        ci = col[m] if col.ndim == 2 else col
        buf[yi, xi] = np.minimum(255.0, buf[yi, xi] + ci * alpha)

    # 1. Yıldızlar
    twinkle = 0.55 + 0.45 * np.sin(t * star_sp + star_ph)
    add_glow(star_x, star_y, star_col * twinkle[:, None], alpha=0.95)

    # Büyüme ve Açılma Zamanlaması
    g = smooth(t / 3.4)                     # Sap büyümesi (0 -> 3.4 sn)
    bloom_t = smooth((t - 3.6) / 7.8)       # Gül açılması (3.6 -> 11.4 sn)
    ring_a = smooth((t - 2.6) / 3.0)        # Geometrik halkaların belirmesi
    sway = math.sin(t * 0.95) * 9.0 * g     # Hafif rüzgar salınımı

    def stem_xy(s):
        x = GX + 13.0 * np.sin(s * math.pi * 1.08) + sway * (s ** 1.6)
        y = GY - s * STEM_H
        return x, y

    tip_x, tip_y = stem_xy(g)
    bloom_cx, bloom_cy = stem_xy(1.0)
    bloom_cy -= 26.0

    # 2. Arka Plan Işıltılı Geometrik Halkalar
    if ring_a > 0.005:
        rot = t * 0.14
        R_big = 225.0 * (0.88 + 0.12 * ring_a)
        ring_col = np.array([212.0, 224.0, 250.0], dtype=np.float32)

        ox = bloom_cx + R_big * np.cos(OUTER_RING_PTS + rot)
        oy = bloom_cy + R_big * np.sin(OUTER_RING_PTS + rot)
        add_glow(ox, oy, ring_col, alpha=0.62 * ring_a)

        R_sub = R_big * 0.52
        for k in range(6):
            ang = k * (math.tau / 6.0) - rot * 0.75
            cx = bloom_cx + R_sub * 0.92 * math.cos(ang)
            cy = bloom_cy + R_sub * 0.92 * math.sin(ang)
            rx = cx + R_sub * np.cos(RING_PTS + rot)
            ry = cy + R_sub * np.sin(RING_PTS + rot)
            add_glow(rx, ry, ring_col, alpha=0.42 * ring_a)

    # 3. Toprak Tepeciği
    add(so_dx + GX, so_y, -500.0, so_col * 0.95)

    # 4. Sap
    m_st = st_s <= g
    if np.any(m_st):
        sx_c, sy_c = stem_xy(st_s[m_st])
        add(sx_c + st_dx[m_st], sy_c, -120.0, st_col[m_st])

    # 5. Dikenler
    for th in thorns:
        tg = smooth((g - th["s"]) * 5.0)
        if tg < 0.02:
            continue
        bx, by = stem_xy(th["s"])
        tx = bx + th["side"] * (4.0 + th["u"] * 12.0 * tg)
        ty = by - th["v"] * 4.5 * tg + th["u"] * 3.5 * tg
        add(tx, ty, -115.0, th["col"])

    # 6. Gül Yaprakları
    for lf in leaves:
        lg = smooth((g - lf["s"]) * 2.6)
        if lg < 0.01:
            continue
        bx, by = stem_xy(lf["s"])
        L_cur = lf["L"] * lg
        us, rr, hh = midrib(L_cur, lf["th0"], lf["curl"] * lg, n=36)
        r_u = np.interp(lf["u"], us, rr)
        h_u = np.interp(lf["u"], us, hh)

        dr = np.gradient(rr)
        dh = np.gradient(hh)
        nlen = np.hypot(dr, dh) + 1e-6
        nx = np.interp(lf["u"], us, -dh / nlen)
        ny = np.interp(lf["u"], us, dr / nlen)

        w_off = lf["v"] * lf["w"] * lg
        lx = bx + lf["side"] * (r_u + nx * w_off)
        ly = by - (h_u + ny * w_off)
        lz = -110.0 + lf["u"] * 8.0
        add(lx, ly, lz, lf["col"])

    # 7. Gül Tomurcuğu ve Açılan Beyaz Gül
    bud_scale = smooth((t - 2.7) / 1.5)
    if bud_scale > 0.01:
        cx, cy = tip_x, tip_y - 4.0 * bud_scale

        # 7a. Çanak Yapraklar (Sepals)
        sepal_open = smooth((bloom_t - 0.02) / 0.55)
        for sp in sepals:
            th0 = lerp(0.18, 1.95, sepal_open)
            curl = lerp(-0.25, 0.80, sepal_open)
            L_sp = 56.0 * bud_scale
            us, rr, hh = midrib(L_sp, th0, curl, n=24)
            r_u = np.interp(sp["u"], us, rr) + 7.0 * bud_scale
            h_u = np.interp(sp["u"], us, hh) - 8.0 * bud_scale

            c_phi = math.cos(sp["phi"])
            s_phi = math.sin(sp["phi"])
            w_off = sp["v"] * sp["w"] * bud_scale

            px = r_u * c_phi - w_off * s_phi
            pz = r_u * s_phi + w_off * c_phi
            py = h_u

            y_rot = py * COS_P - pz * SIN_P
            z_rot = py * SIN_P + pz * COS_P
            add(cx + px, cy - y_rot, z_rot - 25.0, sp["col"])

        # 7b. 26 Katmanlı Beyaz Gül Taç Yaprakları
        for p in petals:
            tier = p["tier"]
            p_open = smooth((bloom_t - p["delay"]) / max(0.25, 1.0 - p["delay"]))
            breathe = 0.024 * math.sin(t * 1.1 + tier * 2.5) * p_open

            th0 = lerp(p["th_closed"], p["th_open"], p_open) + breathe
            curl = lerp(p["curl_closed"], p["curl_open"], p_open)
            L_p = p["L"] * bud_scale * lerp(0.75, 1.0, p_open)

            us, rr, hh = midrib(L_p, th0, curl, n=28)
            u = p["u"]
            v = p["v"]

            r_u = np.interp(u, us, rr) + lerp(3.5, 15.0, tier) * bud_scale
            h_u = np.interp(u, us, hh) - lerp(1.0, 12.0, tier) * bud_scale

            # Gül yaprağının üst dudaktan zarif dışa kıvrımı (recurve)
            recurve = (
                p_open
                * lerp(5.0, 36.0, tier)
                * (np.maximum(0.0, u - 0.56) ** 1.75)
                * (0.55 + 0.45 * (1.0 - v * v))
            )
            r_u = r_u + recurve * bud_scale
            h_u = h_u - recurve * 0.32 * bud_scale

            # Yaprağın kadeh (cup) kavisi
            w_p = p["W_max"] * p["prof"] * p["tip_notch"] * bud_scale * lerp(0.78, 1.0, p_open)
            cup_angle = v * lerp(0.86, 0.58, p_open) * (0.55 + 0.45 * u)
            ang = p["phi"] + cup_angle

            cup_rad = r_u + (1.0 - v * v) * lerp(4.0, 14.0, 1.0 - tier * 0.5) * np.sin(u * math.pi) * bud_scale
            ripple = 2.6 * np.sin(v * 4.2 + p["ripple_phase"]) * (u ** 2.0) * p_open

            px = cup_rad * np.cos(ang) - (v * w_p * 0.36) * np.sin(p["phi"])
            pz = cup_rad * np.sin(ang) + (v * w_p * 0.36) * np.cos(p["phi"])
            py = h_u + ripple

            y_rot = py * COS_P - pz * SIN_P
            z_rot = py * SIN_P + pz * COS_P

            # Derinlik ve üst ay ışığı gölgelendirmesi
            light = np.clip(0.80 + 0.26 * (z_rot / 140.0) + 0.12 * (py / 120.0), 0.56, 1.12)
            col_lit = np.clip(p["col"] * light[:, None], 0.0, 255.0)

            add(cx + px, cy - y_rot, z_rot, col_lit)

    # 8. Uçuşan Ay Işığı Zerreleri
    if bloom_t > 0.15:
        ma = smooth((bloom_t - 0.15) / 0.5)
        drift = ((t * mote_sp + mote_ph * 25.0) % 290.0)
        mx = bloom_cx + mote_r * np.cos(mote_a + t * 0.18) + 12.0 * np.sin(t + mote_ph)
        my = (bloom_cy + 85.0) - drift
        fade = np.sin(np.clip(drift / 290.0, 0.0, 1.0) * math.pi) * ma
        mcol = np.array([238.0, 245.0, 255.0], dtype=np.float32) * fade[:, None]
        add_glow(mx, my, mcol, alpha=0.90)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Beyaz Gül")
    clock = pygame.time.Clock()

    buf = np.zeros((H, W, 3), dtype=np.float32)
    zbuf = np.zeros((H, W), dtype=np.float32)

    t = 0.0
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    t = 0.0  # R tuşu ile animasyonu baştan başlat

        t += min(dt, 0.05)
        render(t, buf, zbuf)

        frame = np.clip(buf, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(screen, np.transpose(frame, (1, 0, 2)))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()