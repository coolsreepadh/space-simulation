"""
================================================================================
 GPU N-BODY ASTROPHYSICS SIMULATOR & UNIVERSE SANDBOX
================================================================================
A single-file, GPU-accelerated N-body gravity sandbox built with Taichi Lang
and its native GGUI (Dear ImGui-backed) system.

Run with:  python main.py
Requires:  pip install taichi numpy
================================================================================
"""

import taichi as ti
import numpy as np
import math
import time

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ==============================================================================
# 1. GLOBAL CONSTANTS
# ==============================================================================
MAX_BODIES      = 4000          # hard cap on simulated massive bodies
STAR_COUNT      = 24000         # background starfield point count
MILKYWAY_COUNT  = 9000          # milky-way band particle count
NEBULA_COUNT    = 6000          # HII / nebula particle count
BUBBLE_COUNT    = 4000          # Fermi/eROSITA bubble particle count
GLOBULAR_COUNT  = 2500          # globular cluster halo particle count
GRID_N          = 48            # spacetime grid resolution (GRID_N x GRID_N)
SKY_RADIUS      = 4000.0

G_CONST   = 1.0                 # simulation-unit gravitational constant
C_LIGHT   = 60.0                # simulation-unit "speed of light" (for relativity demo)

TRAIL_LEN         = 100         # samples kept per trailed object
TRAIL_MAX_BODIES  = 30          # only the first K bodies get orbit trails (perf cap)

vec3 = ti.math.vec3


def au_to_units(au):
    """Compresses real AU distances into compact, ordered sim-space units."""
    return 36.0 * math.sqrt(au)


# ==============================================================================
# 2. FIELDS -- N-BODY STATE
# ==============================================================================
body_pos     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_vel     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_acc     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_prevacc = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_mass    = ti.field(dtype=ti.f32, shape=MAX_BODIES)
body_radius  = ti.field(dtype=ti.f32, shape=MAX_BODIES)
body_temp    = ti.field(dtype=ti.f32, shape=MAX_BODIES)
body_color   = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_active  = ti.field(dtype=ti.i32, shape=MAX_BODIES)
body_kind    = ti.field(dtype=ti.i32, shape=MAX_BODIES)  # 0 normal, 1 disk debris, 2 black hole, 3 doomed star
body_explicit_color = ti.field(dtype=ti.i32, shape=MAX_BODIES)  # 1 = has a fixed RGB, skip blackbody refresh

# Rendering-only warped copies (used when gravitational lensing is enabled)
render_body_pos = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
render_star_pos = ti.Vector.field(3, dtype=ti.f32, shape=STAR_COUNT)
render_mw_pos   = ti.Vector.field(3, dtype=ti.f32, shape=MILKYWAY_COUNT)

# Background decoration fields
star_pos      = ti.Vector.field(3, dtype=ti.f32, shape=STAR_COUNT)
star_color    = ti.Vector.field(3, dtype=ti.f32, shape=STAR_COUNT)
mw_pos        = ti.Vector.field(3, dtype=ti.f32, shape=MILKYWAY_COUNT)
mw_color      = ti.Vector.field(3, dtype=ti.f32, shape=MILKYWAY_COUNT)
nebula_pos    = ti.Vector.field(3, dtype=ti.f32, shape=NEBULA_COUNT)
nebula_color  = ti.Vector.field(3, dtype=ti.f32, shape=NEBULA_COUNT)
bubble_pos    = ti.Vector.field(3, dtype=ti.f32, shape=BUBBLE_COUNT)
bubble_color  = ti.Vector.field(3, dtype=ti.f32, shape=BUBBLE_COUNT)
glob_pos      = ti.Vector.field(3, dtype=ti.f32, shape=GLOBULAR_COUNT)
glob_color    = ti.Vector.field(3, dtype=ti.f32, shape=GLOBULAR_COUNT)

# Spacetime fabric grid (line list: (GRID_N-1)*GRID_N horiz + vert segments *2)
grid_vtx_count = (GRID_N - 1) * GRID_N * 2 * 2
grid_vertices  = ti.Vector.field(3, dtype=ti.f32, shape=grid_vtx_count)

# Constellation line-list buffer (rebuilt once at startup)
CONST_MAX_LINES = 64
const_vertices = ti.Vector.field(3, dtype=ti.f32, shape=CONST_MAX_LINES * 2)

# Center of mass (single-element field so kernels can write it)
com_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

# Orbit trail line buffers (built on the host each frame from small ring buffers)
TRAIL_VTX_PER_OBJ = (TRAIL_LEN - 1) * 2
body_trail_lines    = ti.Vector.field(3, dtype=ti.f32, shape=TRAIL_MAX_BODIES * TRAIL_VTX_PER_OBJ)

_body_trail_buf = np.zeros((TRAIL_MAX_BODIES, TRAIL_LEN, 3), dtype=np.float32)

# ==============================================================================
# 3b. SATURN RING (rendered relative to Saturn's live position)
# ==============================================================================
SATURN_RING_POINTS = 500
_SAT_RING_TILT = math.radians(26.7)
_sat_ring_r = np.random.uniform(2.75, 4.6, SATURN_RING_POINTS)
_sat_ring_ang = np.random.uniform(0, 2 * math.pi, SATURN_RING_POINTS)
_sat_ring_local = np.zeros((SATURN_RING_POINTS, 3), dtype=np.float32)
_z0 = _sat_ring_r * np.sin(_sat_ring_ang)
_sat_ring_local[:, 0] = _sat_ring_r * np.cos(_sat_ring_ang)
_sat_ring_local[:, 1] = -_z0 * math.sin(_SAT_RING_TILT)
_sat_ring_local[:, 2] = _z0 * math.cos(_SAT_RING_TILT)

saturn_ring_pos = ti.Vector.field(3, dtype=ti.f32, shape=SATURN_RING_POINTS)

# ==============================================================================
# 3c. SUN CORONA (a sparse shell of tiny hot points -> reads as a soft glow;
#     GGUI particles are opaque spheres, so a scattered shell is how you fake
#     a halo here). Colors are deliberately > 1.0 so the ambient term alone
#     saturates and the sun never shows a shaded/dark side.
# ==============================================================================
SUN_GLOW_POINTS = 20000
_glow_dir = np.random.normal(size=(SUN_GLOW_POINTS, 3))
_glow_dir /= np.linalg.norm(_glow_dir, axis=1, keepdims=True)
# bias the shell toward the inner edge so density falls off outward; the shell
# starts clear of the 6.5-unit photosphere so it haloes the disc instead of
# speckling it
_glow_frac = np.random.uniform(size=SUN_GLOW_POINTS) ** 2.6
_glow_r = 6.7 + _glow_frac * 8.5
_sun_glow_local = (_glow_dir * _glow_r[:, None]).astype(np.float32)

_glow_t = _glow_frac  # 0 at the inner halo edge -> 1 at the outer edge
_falloff = (1.0 - _glow_t) ** 1.8
_glow_rgb = np.empty((SUN_GLOW_POINTS, 3), dtype=np.float32)
_glow_rgb[:, 0] = 3.0 * _falloff
_glow_rgb[:, 1] = 1.7 * _falloff
_glow_rgb[:, 2] = 0.7 * _falloff
_sun_glow_radius = (0.13 * (1.0 - 0.6 * _glow_t)).astype(np.float32)

sun_glow_pos = ti.Vector.field(3, dtype=ti.f32, shape=SUN_GLOW_POINTS)
sun_glow_color = ti.Vector.field(3, dtype=ti.f32, shape=SUN_GLOW_POINTS)
sun_glow_radius = ti.field(dtype=ti.f32, shape=SUN_GLOW_POINTS)
sun_glow_color.from_numpy(_glow_rgb)
sun_glow_radius.from_numpy(_sun_glow_radius)


# ==============================================================================
# 4. UTILITY FUNCTIONS (BLACKBODY COLOR, ETC.)
# ==============================================================================
@ti.func
def temp_to_rgb(t: ti.f32) -> ti.math.vec3:
    res = ti.math.vec3(0.0, 0.0, 0.0)
    if t > 0.0:
        tt = ti.max(1000.0, ti.min(t, 40000.0)) / 100.0
        r = 255.0
        g = 255.0
        b = 255.0

        if tt <= 66.0:
            r = 255.0
            g = 99.4708025861 * ti.log(tt) - 161.1195681661
            if tt <= 19.0:
                b = 0.0
            else:
                b = 138.5177312231 * ti.log(tt - 10.0) - 305.0447927307
        else:
            r = 329.698727446 * ti.pow(tt - 60.0, -0.1332047592)
            g = 288.1221695283 * ti.pow(tt - 60.0, -0.0755148492)
            b = 255.0

        r = ti.max(0.0, ti.min(r, 255.0)) / 255.0
        g = ti.max(0.0, ti.min(g, 255.0)) / 255.0
        b = ti.max(0.0, ti.min(b, 255.0)) / 255.0
        res = ti.math.vec3(r, g, b)
    return res


@ti.kernel
def refresh_body_colors(n: ti.i32):
    for i in range(n):
        if body_active[i] == 1 and body_explicit_color[i] == 0:
            body_color[i] = temp_to_rgb(body_temp[i])


@ti.kernel
def set_single_body_color(idx: ti.i32, t: ti.f32):
    body_color[idx] = temp_to_rgb(t)
    body_explicit_color[idx] = 0


@ti.kernel
def set_single_body_rgb(idx: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    body_color[idx] = vec3(r, g, b)
    body_explicit_color[idx] = 1


# ==============================================================================
# 5. PHYSICS KERNELS -- VELOCITY VERLET N-BODY INTEGRATION
# ==============================================================================
@ti.kernel
def compute_accelerations(n: ti.i32, eps: ti.f32):
    for i in range(n):
        if body_active[i] == 1:
            a = vec3(0.0, 0.0, 0.0)
            for j in range(n):
                if j != i and body_active[j] == 1:
                    r = body_pos[j] - body_pos[i]
                    dist2 = r.dot(r) + eps * eps
                    inv_d3 = dist2 ** (-1.5)
                    a += G_CONST * body_mass[j] * r * inv_d3
            body_acc[i] = a


@ti.kernel
def verlet_step1(n: ti.i32, dt: ti.f32):
    for i in range(n):
        if body_active[i] == 1:
            body_prevacc[i] = body_acc[i]
            body_pos[i] += body_vel[i] * dt + 0.5 * body_acc[i] * dt * dt


@ti.kernel
def verlet_step2(n: ti.i32, dt: ti.f32):
    for i in range(n):
        if body_active[i] == 1:
            body_vel[i] += 0.5 * (body_prevacc[i] + body_acc[i]) * dt


@ti.kernel
def compute_center_of_mass(n: ti.i32):
    total_mass = 0.0
    weighted = vec3(0.0, 0.0, 0.0)
    for i in range(n):
        if body_active[i] == 1:
            total_mass += body_mass[i]
            weighted += body_pos[i] * body_mass[i]
    if total_mass > 1e-9:
        com_pos[None] = weighted / total_mass


# ==============================================================================
# 6. RELATIVITY HELPERS -- SCHWARZSCHILD TIME DILATION APPROXIMATION
# ==============================================================================
def gravitational_time_dilation(observer_idx, n_np_pos, n_np_mass, n_active):
    if observer_idx is None or observer_idx >= len(n_active) or n_active[observer_idx] == 0:
        return 1.0, None
    p = n_np_pos[observer_idx]
    strongest_factor = 1.0
    strongest_src = None
    for j in range(len(n_active)):
        if j == observer_idx or n_active[j] == 0:
            continue
        r = np.linalg.norm(n_np_pos[j] - p)
        if r < 1e-4:
            continue
        rs = 2.0 * G_CONST * n_np_mass[j] / (C_LIGHT ** 2)
        term = 1.0 - rs / max(r, rs * 1.001)
        term = max(term, 1e-6)
        factor = math.sqrt(term)
        if factor < strongest_factor:
            strongest_factor = factor
            strongest_src = j
    return strongest_factor, strongest_src


def schwarzschild_radius(mass):
    return 2.0 * G_CONST * mass / (C_LIGHT * C_LIGHT)


# ==============================================================================
# 7. GRAVITATIONAL LENSING -- "REVEALS THE INVISIBLE BLACK HOLE" (Interstellar-style)
# ==============================================================================
@ti.kernel
def compute_body_lensing(n: ti.i32,
                          cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                          bh_x: ti.f32, bh_y: ti.f32, bh_z: ti.f32, bh_rs: ti.f32):
    cam = vec3(cam_x, cam_y, cam_z)
    bh = vec3(bh_x, bh_y, bh_z)
    for i in range(n):
        out = body_pos[i]
        if body_active[i] == 1:
            p = body_pos[i]
            view = p - cam
            view_len = ti.sqrt(view.dot(view))
            if view_len > 1e-5:
                dir_cp = view / view_len
                bh_rel = bh - cam
                t = bh_rel.dot(dir_cp)
                if t > 0.0:
                    closest = cam + dir_cp * t
                    bvec = closest - bh
                    b = ti.sqrt(bvec.dot(bvec))
                    if b > bh_rs * 0.05:
                        warp_mag = (bh_rs * bh_rs * 1.5) / (b + 0.2 * bh_rs)
                        warp_mag = ti.min(warp_mag, bh_rs * 8.0)
                        deflect = bvec / b
                        behind_w = ti.min(ti.max(t / (t + 60.0), 0.0), 1.0)
                        out = p + deflect * warp_mag * behind_w
        render_body_pos[i] = out


@ti.kernel
def compute_star_lensing(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                          bh_x: ti.f32, bh_y: ti.f32, bh_z: ti.f32, bh_rs: ti.f32):
    cam = vec3(cam_x, cam_y, cam_z)
    bh = vec3(bh_x, bh_y, bh_z)
    for i in star_pos:
        p = star_pos[i]
        out = p
        view = p - cam
        view_len = ti.sqrt(view.dot(view))
        if view_len > 1e-5:
            dir_cp = view / view_len
            bh_rel = bh - cam
            t = bh_rel.dot(dir_cp)
            if t > 0.0:
                closest = cam + dir_cp * t
                bvec = closest - bh
                b = ti.sqrt(bvec.dot(bvec))
                if b > bh_rs * 0.05:
                    warp_mag = (bh_rs * bh_rs * 1.5) / (b + 0.2 * bh_rs)
                    warp_mag = ti.min(warp_mag, bh_rs * 8.0)
                    deflect = bvec / b
                    behind_w = ti.min(ti.max(t / (t + 60.0), 0.0), 1.0)
                    out = p + deflect * warp_mag * behind_w
        render_star_pos[i] = out


@ti.kernel
def compute_mw_lensing(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                        bh_x: ti.f32, bh_y: ti.f32, bh_z: ti.f32, bh_rs: ti.f32):
    cam = vec3(cam_x, cam_y, cam_z)
    bh = vec3(bh_x, bh_y, bh_z)
    for i in mw_pos:
        p = mw_pos[i]
        out = p
        view = p - cam
        view_len = ti.sqrt(view.dot(view))
        if view_len > 1e-5:
            dir_cp = view / view_len
            bh_rel = bh - cam
            t = bh_rel.dot(dir_cp)
            if t > 0.0:
                closest = cam + dir_cp * t
                bvec = closest - bh
                b = ti.sqrt(bvec.dot(bvec))
                if b > bh_rs * 0.05:
                    warp_mag = (bh_rs * bh_rs * 1.5) / (b + 0.2 * bh_rs)
                    warp_mag = ti.min(warp_mag, bh_rs * 8.0)
                    deflect = bvec / b
                    behind_w = ti.min(ti.max(t / (t + 60.0), 0.0), 1.0)
                    out = p + deflect * warp_mag * behind_w
        render_mw_pos[i] = out


# ==============================================================================
# 8. TIDAL DISRUPTION -- "DESTROY A STAR TO FEED THE ACCRETION DISK"
# ==============================================================================
TIDAL_SHED_PER_FRAME = 0.9
STAR_MIN_MASS = 4.0


def process_tidal_disruption(sim):
    if sim.bh_idx < 0 or sim.star_idx < 0:
        return
    if sim.star_idx >= sim.num_bodies or body_active[sim.star_idx] == 0:
        return

    bp = body_pos[sim.bh_idx]
    bm = body_mass[sim.bh_idx]
    sp = body_pos[sim.star_idx]
    sm = body_mass[sim.star_idx]
    sr = body_radius[sim.star_idx]

    dx, dy, dz = sp.x - bp.x, sp.y - bp.y, sp.z - bp.z
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r < 1e-4 or sm <= STAR_MIN_MASS:
        return

    tidal_r = sr * (bm / max(sm, 1e-3)) ** (1.0 / 3.0) * 4.0
    if r >= tidal_r or sim.num_bodies >= MAX_BODIES - 4:
        return

    shed = min(TIDAL_SHED_PER_FRAME, sm - STAR_MIN_MASS)
    if shed <= 0.0:
        return

    new_mass = sm - shed
    new_radius = sr * max(new_mass / sm, 0.05) ** (1.0 / 3.0)
    body_mass[sim.star_idx] = new_mass
    body_radius[sim.star_idx] = new_radius

    for _ in range(2):
        idx = sim.num_bodies
        frac = np.random.uniform(0.35, 1.0)
        drop_r = max(r * frac, tidal_r * 0.3)
        ang = math.atan2(dz, dx) + np.random.uniform(-0.5, 0.5)
        speed = math.sqrt(G_CONST * bm / max(drop_r, 1.0)) * np.random.uniform(0.9, 1.05)
        px = bp.x + drop_r * math.cos(ang)
        pz = bp.z + drop_r * math.sin(ang)
        py = bp.y + np.random.uniform(-1.0, 1.0)
        vx = -speed * math.sin(ang)
        vz = speed * math.cos(ang)
        disk_temp = 9000.0 + 16000.0 * (1.0 - min(drop_r, 100.0) / 100.0)

        body_pos[idx] = ti.Vector([float(px), float(py), float(pz)])
        body_vel[idx] = ti.Vector([float(vx), 0.0, float(vz)])
        body_mass[idx] = float(shed) / 2.0
        body_radius[idx] = 0.5
        body_temp[idx] = float(disk_temp)
        body_kind[idx] = 1
        body_active[idx] = 1
        set_single_body_color(idx, float(disk_temp))
        sim.num_bodies += 1

    if new_mass <= STAR_MIN_MASS:
        body_active[sim.star_idx] = 0
        sim.star_idx = -1


# ==============================================================================
# 9. SPACETIME FABRIC GRID
# ==============================================================================
@ti.kernel
def compute_spacetime_grid(n: ti.i32, center_x: ti.f32, center_z: ti.f32,
                            plane_extent: ti.f32, depth_scale: ti.f32, eps: ti.f32):
    half = plane_extent * 0.5
    step = plane_extent / (GRID_N - 1)
    # horizontal lines (constant z, varying x) -- grid follows the camera so it never shows an edge
    for gz in range(GRID_N):
        for gx in range(GRID_N - 1):
            for k in ti.static(range(2)):
                gxx = gx + k
                x = center_x - half + gxx * step
                z = center_z - half + gz * step
                y = 0.0
                for b in range(n):
                    if body_active[b] == 1:
                        dx = x - body_pos[b].x
                        dz = z - body_pos[b].z
                        dist2 = dx * dx + dz * dz + eps * eps
                        y -= depth_scale * body_mass[b] / ti.sqrt(dist2)
                lidx = (gz * (GRID_N - 1) + gx) * 2 + k
                grid_vertices[lidx] = vec3(x, y, z)
    offset = (GRID_N - 1) * GRID_N * 2
    # vertical lines (constant x, varying z)
    for gx in range(GRID_N):
        for gz in range(GRID_N - 1):
            for k in ti.static(range(2)):
                gzz = gz + k
                x = center_x - half + gx * step
                z = center_z - half + gzz * step
                y = 0.0
                for b in range(n):
                    if body_active[b] == 1:
                        dx = x - body_pos[b].x
                        dz = z - body_pos[b].z
                        dist2 = dx * dx + dz * dz + eps * eps
                        y -= depth_scale * body_mass[b] / ti.sqrt(dist2)
                lidx = offset + (gx * (GRID_N - 1) + gz) * 2 + k
                grid_vertices[lidx] = vec3(x, y, z)


# ==============================================================================
# 10. PROCEDURAL BACKGROUND -- SKYBOX, MILKY WAY BAND, DEEP-SPACE PHENOMENA
# ==============================================================================
@ti.kernel
def init_starfield(seed: ti.i32):
    for i in range(STAR_COUNT):
        u = ti.random()
        v = ti.random()
        theta = 2.0 * math.pi * u
        phi = ti.acos(2.0 * v - 1.0)
        r = SKY_RADIUS * (0.85 + 0.15 * ti.random())
        x = r * ti.sin(phi) * ti.cos(theta)
        y = r * ti.sin(phi) * ti.sin(theta)
        z = r * ti.cos(phi)
        star_pos[i] = vec3(x, y, z)
        t = 3000.0 + ti.random() * 20000.0
        c = temp_to_rgb(t)
        brightness = 0.6 + 0.4 * ti.random()
        star_color[i] = c * brightness


@ti.kernel
def init_milkyway():
    for i in range(MILKYWAY_COUNT):
        theta = ti.random() * 2.0 * math.pi
        band_scatter = (ti.random() - 0.5) * 0.18
        r = SKY_RADIUS * (0.9 + 0.1 * ti.random())
        x = r * ti.cos(theta)
        z = r * ti.sin(theta)
        y = r * band_scatter
        mw_pos[i] = vec3(x, y, z)
        t = 4000.0 + ti.random() * 6000.0
        c = temp_to_rgb(t)
        mw_color[i] = c * (0.15 + 0.25 * ti.random())


@ti.kernel
def init_nebula(center_x: ti.f32, center_y: ti.f32, center_z: ti.f32, spread: ti.f32):
    for i in range(NEBULA_COUNT):
        d = ti.Vector([ti.randn(), ti.randn(), ti.randn()]) * spread
        nebula_pos[i] = vec3(center_x, center_y, center_z) + d
        mix = ti.random()
        col = vec3(1.0, 0.25, 0.35) if mix > 0.25 else vec3(0.25, 0.85, 0.95)
        nebula_color[i] = col * (0.35 + 0.4 * ti.random())


@ti.kernel
def init_bubbles(cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32, height: ti.f32):
    for i in range(BUBBLE_COUNT):
        u = ti.random()
        ang = ti.random() * 2.0 * math.pi
        rad = radius * ti.sqrt(ti.random())
        lobe_y = height * (0.15 + 0.85 * ti.random())
        sign = 1.0 if i % 2 == 0 else -1.0
        x = cx + rad * ti.cos(ang) * (1.0 - u * 0.4)
        z = cz + rad * ti.sin(ang) * (1.0 - u * 0.4)
        y = cy + sign * lobe_y
        bubble_pos[i] = vec3(x, y, z)
        bubble_color[i] = vec3(0.65, 0.15, 0.85) * (0.25 + 0.35 * ti.random())


@ti.kernel
def init_globular(cx: ti.f32, cy: ti.f32, cz: ti.f32, radius: ti.f32):
    for i in range(GLOBULAR_COUNT):
        d = ti.Vector([ti.randn(), ti.randn(), ti.randn()])
        d = d.normalized() * radius * ti.pow(ti.random(), 0.33)
        glob_pos[i] = vec3(cx, cy, cz) + d
        t = 4500.0 + ti.random() * 2000.0
        glob_color[i] = temp_to_rgb(t) * (0.5 + 0.4 * ti.random())


# ==============================================================================
# 11. HARD-CODED CONSTELLATION CATALOG
# ==============================================================================
CONSTELLATIONS = {
    "Ursa Major (Big Dipper)": {
        "stars_radec_deg": [
            (165.9, 61.8), (183.9, 57.0), (165.5, 56.4),
            (183.9, 53.9), (193.5, 55.0), (200.9, 54.9), (206.9, 49.3),
        ],
        "lines": [(0, 2), (2, 3), (3, 1), (1, 0), (3, 4), (4, 5), (5, 6)],
    },
    "Orion": {
        "stars_radec_deg": [
            (88.8, 7.4), (81.3, 6.3), (84.0, -1.2),
            (85.2, -1.9), (86.9, -1.2), (78.6, -8.2), (86.9, -9.7),
        ],
        "lines": [(0, 3), (3, 5), (1, 3), (3, 6), (2, 4), (4, 3)],
    },
    "Cassiopeia": {
        "stars_radec_deg": [
            (2.3, 59.1), (10.1, 56.5), (14.2, 60.7), (21.5, 60.2), (28.6, 63.7),
        ],
        "lines": [(0, 1), (1, 2), (2, 3), (3, 4)],
    },
}


def build_constellation_lines():
    verts = np.zeros((CONST_MAX_LINES * 2, 3), dtype=np.float32)
    slot = 0
    for name, data in CONSTELLATIONS.items():
        pts = []
        for ra_deg, dec_deg in data["stars_radec_deg"]:
            ra = math.radians(ra_deg)
            dec = math.radians(dec_deg)
            r = SKY_RADIUS * 0.98
            x = r * math.cos(dec) * math.cos(ra)
            y = r * math.sin(dec)
            z = r * math.cos(dec) * math.sin(ra)
            pts.append((x, y, z))
        for (a, b) in data["lines"]:
            if slot >= CONST_MAX_LINES:
                break
            verts[slot * 2 + 0] = pts[a]
            verts[slot * 2 + 1] = pts[b]
            slot += 1
    const_vertices.from_numpy(verts)
    return slot


# ==============================================================================
# 12. SIMULATION PRESETS
# ==============================================================================
class SimState:
    def __init__(self):
        self.num_bodies = 0
        self.selected = 0
        self.follow_com = False
        self.follow_body = -1
        self.bh_idx = -1
        self.star_idx = -1
        self.bh_candidates = []
        self.body_names = []
        self.star_initial_mass = 0.0
        self.saturn_idx = -1
        self.sol_idx = -1


def load_arrays(pos, vel, mass, radius, temp, colors=None, kinds=None):
    n = len(mass)
    assert n <= MAX_BODIES, "preset exceeds MAX_BODIES"
    full_pos = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_vel = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_mass = np.zeros(MAX_BODIES, dtype=np.float32)
    full_radius = np.zeros(MAX_BODIES, dtype=np.float32)
    full_temp = np.zeros(MAX_BODIES, dtype=np.float32)
    full_active = np.zeros(MAX_BODIES, dtype=np.int32)
    full_kind = np.zeros(MAX_BODIES, dtype=np.int32)

    full_pos[:n] = pos
    full_vel[:n] = vel
    full_mass[:n] = mass
    full_radius[:n] = radius
    full_temp[:n] = temp
    full_active[:n] = 1
    if kinds is not None:
        full_kind[:n] = kinds

    body_pos.from_numpy(full_pos)
    body_vel.from_numpy(full_vel)
    body_mass.from_numpy(full_mass)
    body_radius.from_numpy(full_radius)
    body_temp.from_numpy(full_temp)
    body_active.from_numpy(full_active)
    body_kind.from_numpy(full_kind)
    body_explicit_color.from_numpy(np.zeros(MAX_BODIES, dtype=np.int32))
    refresh_body_colors(n)

    if colors is not None:
        for i, c in enumerate(colors):
            if c is not None:
                set_single_body_rgb(i, float(c[0]), float(c[1]), float(c[2]))

    return n


def preset_solar_system():
    """Inner rocky planets + outer gas/ice giants, slightly-to-scale (compressed, spaced out)."""
    # name, mass, radius, temp(None => explicit color), AU, phase, explicit_color
    bodies = [
        # Sol's color is deliberately over-bright (> 1.0): GGUI shades every
        # particle, so pushing the color past saturation makes even the
        # ambient-only "night" side clamp to white -- i.e. the sun reads as a
        # self-luminous source instead of a half-lit ball with a dark side.
        ("Sol",     3000.0, 6.5,  None, 0.0,   0.00, (4.5, 4.2, 3.8)),
        ("Mercury", 0.055,  0.45, None, 0.39,  0.35, (0.62, 0.58, 0.55)),
        ("Venus",   0.815,  0.95, None, 0.72,  1.85, (0.92, 0.78, 0.55)),
        ("Earth",   1.0,    1.00, None, 1.00,  2.90, (0.30, 0.55, 0.95)),
        ("Mars",    0.107,  0.70, None, 1.52,  4.20, (0.80, 0.40, 0.25)),
        ("Jupiter", 11.0,   2.40, None, 5.20,  0.70, (0.85, 0.70, 0.50)),
        ("Saturn",  9.0,    2.10, None, 9.58,  2.10, (0.90, 0.82, 0.60)),
        ("Uranus",  4.0,    1.50, None, 19.20, 3.60, (0.60, 0.85, 0.90)),
        ("Neptune", 4.0,    1.40, None, 30.05, 5.10, (0.25, 0.35, 0.90)),
    ]
    pos, vel, mass, rad, temp, colors, names, kinds = [], [], [], [], [], [], [], []
    sun_mass = bodies[0][1]
    for (name, m, r, t, au, phase, col) in bodies:
        if au == 0.0:
            pos.append([0, 0, 0]); vel.append([0, 0, 0])
        else:
            orbit_r = au_to_units(au)
            x = orbit_r * math.cos(phase)
            z = orbit_r * math.sin(phase)
            speed = math.sqrt(G_CONST * sun_mass / orbit_r)
            vx = -speed * math.sin(phase)
            vz = speed * math.cos(phase)
            pos.append([x, 0, z]); vel.append([vx, 0, vz])
        mass.append(m); rad.append(r); temp.append(t if t is not None else 0.0)
        colors.append(col); names.append(name); kinds.append(0)
    return (np.array(pos), np.array(vel), np.array(mass),
            np.array(rad), np.array(temp), colors, names, kinds)


def preset_spacetime_demo():
    pos = [[0, 0, 0]]
    vel = [[0, 0, 0]]
    mass = [8000.0]
    rad = [4.0]
    temp = [0.0]
    names = ["Central Mass"]
    kinds = [2]
    for i in range(6):
        orbit_r = 40.0 + i * 18.0
        phase = i * 1.05
        x = orbit_r * math.cos(phase)
        z = orbit_r * math.sin(phase)
        speed = math.sqrt(G_CONST * mass[0] / orbit_r)
        vx = -speed * math.sin(phase)
        vz = speed * math.cos(phase)
        pos.append([x, 0, z]); vel.append([vx, 0, vz])
        mass.append(0.4); rad.append(0.6); temp.append(6000 + i * 400)
        names.append(""); kinds.append(0)
    return (np.array(pos), np.array(vel), np.array(mass),
            np.array(rad), np.array(temp), None, names, kinds)


def _gas_palette_color(radial_frac):
    """Blue/purple/magenta gas-density style palette (not blackbody) for the merger view."""
    core = np.array([1.00, 0.85, 0.95])
    mid = np.array([0.85, 0.20, 0.85])
    outer = np.array([0.12, 0.12, 0.55])
    radial_frac = min(max(radial_frac, 0.0), 1.0)
    if radial_frac < 0.5:
        t = radial_frac / 0.5
        c = core * (1.0 - t) + mid * t
    else:
        t = (radial_frac - 0.5) / 0.5
        c = mid * (1.0 - t) + outer * t
    c = c * (0.55 + 0.45 * np.random.uniform())
    return (float(c[0]), float(c[1]), float(c[2]))


def preset_galaxy_merger():
    """Two spiral galaxies on a collision course, gas-look coloring, tidal tails emerge as they merge.
    No black hole dot -- just a small bright nucleus, matching a Toomre-style gas/star merger render."""
    def spiral_disk(center, com_vel, core_mass, n_stars, radius_max, spin_sign, core_name, n_arms=2):
        p, v, m, r, t, c, nm, kd = [], [], [], [], [], [], [], []
        p.append(list(center)); v.append(list(com_vel))
        m.append(core_mass); r.append(0.6); t.append(0.0)
        c.append((1.0, 0.95, 0.98)); nm.append(core_name); kd.append(0)
        for i in range(n_stars):
            rr = radius_max * (0.08 + 0.92 * np.random.uniform() ** 0.5)
            arm_phase = 3.0 * math.log(rr + 1.0)
            arm_idx = np.random.randint(0, n_arms)
            arm_offset = (2.0 * math.pi / n_arms) * arm_idx
            scatter = np.random.normal(0, 0.30)
            ang = arm_phase + arm_offset + scatter
            x = rr * math.cos(ang)
            z = rr * math.sin(ang)
            y = np.random.normal(0, 1.2)
            speed = spin_sign * math.sqrt(G_CONST * core_mass / max(rr, 1.0)) * 0.9
            vx = -speed * math.sin(ang)
            vz = speed * math.cos(ang)
            p.append([center[0] + x, center[1] + y, center[2] + z])
            v.append([com_vel[0] + vx, com_vel[1], com_vel[2] + vz])
            m.append(0.03)
            r.append(0.25)
            t.append(0.0)
            c.append(_gas_palette_color(rr / radius_max))
            nm.append(""); kd.append(0)
        return p, v, m, r, t, c, nm, kd

    n_each = (MAX_BODIES - 2) // 2
    p1, v1, m1, r1, t1, c1, nm1, kd1 = spiral_disk((-90, 0, 0), (2.2, 0, 0.4), 1400.0, n_each, 70.0, 1.0, "Core-A")
    p2, v2, m2, r2, t2, c2, nm2, kd2 = spiral_disk((90, 10, 0), (-2.2, 0, -0.4), 1100.0, n_each, 60.0, -1.0, "Core-B")

    pos = np.array(p1 + p2)
    vel = np.array(v1 + v2)
    mass = np.array(m1 + m2)
    rad = np.array(r1 + r2)
    temp = np.array(t1 + t2)
    colors = c1 + c2
    names = nm1 + nm2
    kinds = kd1 + kd2
    return pos, vel, mass, rad, temp, colors, names, kinds


def preset_gargantua_black_hole():
    """Supermassive black hole with a hot accretion disk plus a doomed star on a decaying,
    tidally-disruptive plunge orbit -- it gets shredded over successive close passes, feeding
    the disk (see 'Black Hole FX' panel)."""
    pos = [[0.0, 0.0, 0.0]]
    vel = [[0.0, 0.0, 0.0]]
    mass = [25000.0]
    rad = [8.0]
    temp = [0.0]
    names = ["Gargantua"]
    kinds = [2]

    n_disk = 2000
    for i in range(n_disk):
        r = 15.0 + 85.0 * (i / n_disk) ** 0.5
        phase = np.random.uniform(0, 2 * math.pi)
        y_offset = np.random.normal(0, 0.16)

        x = r * math.cos(phase)
        z = r * math.sin(phase)

        speed = math.sqrt(G_CONST * mass[0] / r)
        vx = -speed * math.sin(phase)
        vz = speed * math.cos(phase)

        pos.append([x, y_offset, z])
        vel.append([vx, 0.0, vz])
        mass.append(0.01)
        rad.append(0.55)
        temp.append(9000 + 20000 * (1.0 - r / 100.0))
        names.append(""); kinds.append(1)

    # Doomed star: sub-circular speed -> eccentric orbit that plunges inside the tidal
    # radius on every periapsis, gradually shredding into the disk.
    star_orbit_r = 55.0
    star_phase = 0.9
    speed = math.sqrt(G_CONST * mass[0] / star_orbit_r) * 0.82
    sx = star_orbit_r * math.cos(star_phase)
    sz = star_orbit_r * math.sin(star_phase)
    svx = -speed * math.sin(star_phase)
    svz = speed * math.cos(star_phase)
    pos.append([sx, 3.0, sz])
    vel.append([svx, 0.0, svz])
    mass.append(140.0)
    rad.append(3.2)
    temp.append(8200.0)
    names.append("Doomed Star")
    kinds.append(3)

    return (np.array(pos), np.array(vel), np.array(mass), np.array(rad), np.array(temp),
            None, names, kinds)


PRESET_FUNCS = {
    "1) The Solar System": preset_solar_system,
    "2) Spacetime Curvature Demo": preset_spacetime_demo,
    "3) Galaxy Merger (Gas Look)": preset_galaxy_merger,
    "4) Gargantua Black Hole (Interstellar)": preset_gargantua_black_hole,
}


# ==============================================================================
# 13. ORBIT TRAILS (host-side ring buffers -> flat line-pair buffers for scene.lines)
# ==============================================================================
def _update_trail_lines(buf, positions, count):
    """buf: (K, TRAIL_LEN, 3) ring buffer. positions: (count, 3) latest samples."""
    buf[:] = np.roll(buf, -1, axis=1)
    buf[:count, -1, :] = positions[:count]
    if count < buf.shape[0]:
        buf[count:, -1, :] = buf[count:, -2, :]


def _trail_buf_to_lines(buf):
    segs_a = buf[:, :-1, :]
    segs_b = buf[:, 1:, :]
    k, l1, _ = segs_a.shape
    interleaved = np.empty((k, l1 * 2, 3), dtype=np.float32)
    interleaved[:, 0::2, :] = segs_a
    interleaved[:, 1::2, :] = segs_b
    return interleaved.reshape(-1, 3)


# ==============================================================================
# 14. MAIN APPLICATION
# ==============================================================================
def main():
    sim = SimState()

    window = ti.ui.Window("GPU N-Body Universe Sandbox", (1600, 900), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.01, 0.01, 0.02))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0, 60, 220)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)
    # The default far-clip plane is far too short for a scene with a starfield
    # out at SKY_RADIUS=4000 -- without this every star and the whole Milky
    # Way band are silently clipped and the background renders pure black.
    camera.z_near(0.1)
    camera.z_far(6000.0)
    gui = window.get_gui()

    # -------- one-time background procedural generation --------
    init_starfield(1)
    init_milkyway()
    init_nebula(60.0, 5.0, -30.0, 14.0)
    init_bubbles(0.0, 0.0, 0.0, 25.0, 90.0)
    init_globular(-150.0, 40.0, 60.0, 18.0)
    build_constellation_lines()

    # Placeholder plain assignment so `load_preset_by_label` (defined further
    # down) can bind it as nonlocal; the actual scene load happens further
    # below via load_preset_by_label("1) The Solar System"), once that
    # function -- and the camera it repositions -- both exist.
    current_preset = "1) The Solar System"

    # -------- simulation control state --------
    paused = False
    reverse_time = False
    time_scale = 1.0
    base_dt = 0.02
    epsilon = 1.0
    substeps = 2
    target_fps = 60
    fps_unlocked = False
    fps_options = [30, 60, 120]

    show_stars = True
    show_milkyway = True
    show_constellations = True
    show_nebula = False
    show_bubbles = False
    show_globular = False
    show_spacetime_grid = False
    grid_depth_scale = 0.02
    grid_extent = 900.0  # re-centered on the camera each frame -> reads as an infinite grid

    star_quality = 1.0
    nebula_quality = 1.0

    spawn_mass = 1.0
    spawn_radius = 1.0
    spawn_temp = 6000.0
    drag_active = False
    drag_start_pos = None
    drag_spawn_point = None

    lensing_enabled = False
    lens_strength = 1.0

    show_trails = True

    frame_step_requested = False
    last_frame_time = time.perf_counter()

    n_lines_const = min(len(sum([d["lines"] for d in CONSTELLATIONS.values()], [])), CONST_MAX_LINES)

    _body_trail_buf[:] = 0.0

    _prev_keys = {}

    def key_edge(k):
        """True on the frame a key transitions from up to down (no repeats while held)."""
        down = window.is_pressed(k)
        edge = down and not _prev_keys.get(k, False)
        _prev_keys[k] = down
        return edge

    def load_preset_by_label(label):
        nonlocal current_preset, paused, epsilon
        nonlocal show_stars, show_milkyway, show_constellations, lensing_enabled, show_trails
        func = PRESET_FUNCS[label]
        result = func()
        pp, vv, mm, rr, tt, colors, names, kinds = result
        sim.num_bodies = load_arrays(pp, vv, mm, rr, tt, colors, kinds)
        sim.body_names = names
        sim.selected = 0
        sim.follow_body = -1
        sim.follow_com = False
        sim.bh_candidates = [i for i, k in enumerate(kinds) if k == 2]
        sim.bh_idx = sim.bh_candidates[0] if sim.bh_candidates else -1
        sim.star_idx = next((i for i, k in enumerate(kinds) if k == 3), -1)
        sim.star_initial_mass = mm[sim.star_idx] if sim.star_idx >= 0 else 0.0
        sim.saturn_idx = names.index("Saturn") if "Saturn" in names else -1
        sim.sol_idx = names.index("Sol") if "Sol" in names else -1
        current_preset = label

        if label.startswith("3"):
            show_stars, show_milkyway, show_constellations = False, False, False
            lensing_enabled, show_trails = False, False
            epsilon = 4.0
            camera.position(0, 90, 340)
        elif label.startswith("4"):
            show_stars, show_milkyway, show_constellations = True, True, False
            lensing_enabled, show_trails = True, True
            epsilon = 1.0
            camera.position(0, 40, 140)
        elif label.startswith("1"):
            show_stars, show_milkyway, show_constellations = True, True, True
            lensing_enabled, show_trails = False, True
            epsilon = 1.0
            # pulled back far enough that Neptune's orbit (~197 units out) is
            # actually inside the frustum instead of being cropped out of view
            camera.position(0, 150, 520)
        else:
            show_stars, show_milkyway, show_constellations = True, True, True
            lensing_enabled, show_trails = False, False
            epsilon = 1.0
            camera.position(0, 60, 220)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)
        paused = False

    _preset_keys = {"1": None, "2": None, "3": None, "4": None}
    for _label in PRESET_FUNCS:
        _preset_keys[_label[0]] = _label

    load_preset_by_label("1) The Solar System")

    while window.running:
        frame_start = time.perf_counter()

        # ---------------- CAMERA ----------------
        camera.track_user_inputs(window, movement_speed=0.6, hold_key=ti.ui.RMB)
        if sim.follow_body >= 0 and sim.follow_body < sim.num_bodies:
            target = body_pos[sim.follow_body]
            camera.lookat(target.x, target.y, target.z)
        elif sim.follow_com:
            compute_center_of_mass(sim.num_bodies)
            c = com_pos[None]
            camera.lookat(c.x, c.y, c.z)
        scene.set_camera(camera)
        scene.ambient_light((0.34, 0.34, 0.38))
        scene.point_light(pos=(0, 18, 0), color=(1.0, 0.95, 0.85))

        # ================= SILENT KEYBOARD CONTROLS (zero on-screen UI) =================
        #   space = pause/resume        |   1-4   = load preset N        |  [ / ] = time scale
        #   f     = merger face-on view |   g     = merger edge-on view
        if key_edge(ti.ui.SPACE):
            paused = not paused
        if window.is_pressed("]"):
            time_scale = min(time_scale * 1.03, 1000.0)
        if window.is_pressed("["):
            time_scale = max(time_scale / 1.03, 0.001)
        for _k, _label in _preset_keys.items():
            if _label is not None and key_edge(_k):
                load_preset_by_label(_label)
        if current_preset.startswith("3"):
            if key_edge("f"):
                camera.position(0, 240, 0.001)
                camera.lookat(0, 0, 0)
                camera.up(0, 0, 1)
            if key_edge("g"):
                camera.position(240, 10, 0)
                camera.lookat(0, 0, 0)
                camera.up(0, 1, 0)

        # ================= GUI PANELS =================
        with gui.sub_window("Time Suite", 0.01, 0.01, 0.24, 0.30) as w:
            if w.button("Play" if paused else "Pause"):
                paused = not paused
            if w.button("Single Step"):
                frame_step_requested = True
            reverse_time = w.checkbox("Reverse Time Direction", reverse_time)
            time_scale = w.slider_float("Time Scale (x)", time_scale, 0.001, 1000.0)
            epsilon = w.slider_float("Softening (epsilon)", epsilon, 0.05, 10.0)
            substeps = w.slider_int("Substeps/frame", substeps, 1, 8)
            w.text(f"Bodies active: {sim.num_bodies}/{MAX_BODIES}")

        with gui.sub_window("Performance", 0.01, 0.32, 0.24, 0.20) as w:
            fps_unlocked = w.checkbox("Uncapped FPS", fps_unlocked)
            if not fps_unlocked:
                for opt in fps_options:
                    if w.checkbox(f"{opt} FPS", target_fps == opt):
                        target_fps = opt
            star_quality = w.slider_float("Starfield Quality", star_quality, 0.05, 1.0)
            nebula_quality = w.slider_float("Nebula Quality", nebula_quality, 0.05, 1.0)
            w.text(f"Frame time: {(time.perf_counter()-last_frame_time)*1000:.1f} ms")

        with gui.sub_window("Presets", 0.01, 0.53, 0.24, 0.24) as w:
            for label in PRESET_FUNCS:
                if w.button(label):
                    load_preset_by_label(label)
            if current_preset.startswith("3"):
                w.text("Merger views: (or press f / g)")
                if w.button("Face-on View"):
                    camera.position(0, 240, 0.001)
                    camera.lookat(0, 0, 0)
                    camera.up(0, 0, 1)
                if w.button("Edge-on View"):
                    camera.position(240, 10, 0)
                    camera.lookat(0, 0, 0)
                    camera.up(0, 1, 0)

        with gui.sub_window("Visual Toggles", 0.01, 0.78, 0.24, 0.21) as w:
            show_stars = w.checkbox("Starfield Skybox", show_stars)
            show_milkyway = w.checkbox("Milky Way Band", show_milkyway)
            show_constellations = w.checkbox("Constellations", show_constellations)
            show_nebula = w.checkbox("H II Nebula Regions", show_nebula)
            show_bubbles = w.checkbox("Fermi/eROSITA Bubbles", show_bubbles)
            show_globular = w.checkbox("Globular Cluster Halo", show_globular)
            show_spacetime_grid = w.checkbox("Spacetime Fabric Grid", show_spacetime_grid)
            show_trails = w.checkbox("Show Orbit Trails", show_trails)

        with gui.sub_window("Black Hole FX", 0.26, 0.01, 0.24, 0.22) as w:
            lensing_enabled = w.checkbox("Gravitational Lensing (Interstellar)", lensing_enabled)
            lens_strength = w.slider_float("Lens Strength", lens_strength, 0.2, 4.0)
            if sim.bh_idx >= 0 and sim.bh_idx < sim.num_bodies and body_active[sim.bh_idx] == 1:
                w.text(f"Black hole: {sim.body_names[sim.bh_idx] or ('body #' + str(sim.bh_idx))}")
                w.text(f"Mass: {body_mass[sim.bh_idx]:.0f}")
            else:
                w.text("No black hole in this preset.")
            if sim.star_idx >= 0 and sim.star_idx < sim.num_bodies and body_active[sim.star_idx] == 1:
                init_m = max(sim.star_initial_mass, 1e-3)
                pct = 100.0 * (body_mass[sim.star_idx] / init_m)
                w.text(f"Doomed star remaining: {pct:.0f}%")
            elif current_preset.startswith("4"):
                w.text("Star fully consumed into disk.")

        with gui.sub_window("Spawn Tool (Sandbox)", 0.75, 0.01, 0.24, 0.34) as w:
            w.text("Pause sim, then drag in viewport")
            w.text("with LEFT mouse to set velocity.")
            spawn_mass = w.slider_float("Mass", spawn_mass, 0.01, 5000.0)
            spawn_radius = w.slider_float("Radius", spawn_radius, 0.1, 20.0)
            spawn_temp = w.slider_float("Temperature (K)", spawn_temp, 500.0, 30000.0)
            if w.button("Spawn At Camera Focus"):
                fwd = (camera.curr_lookat - camera.curr_position)
                fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
                spawn_point = np.array(camera.curr_position) + fwd * 60.0
                spawn_body(spawn_point, np.array([0.0, 0.0, 0.0]),
                           spawn_mass, spawn_radius, spawn_temp, sim)
            if w.button("Spawn Black Hole At Focus"):
                fwd = (camera.curr_lookat - camera.curr_position)
                fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
                spawn_point = np.array(camera.curr_position) + fwd * 60.0
                spawn_body(spawn_point, np.array([0.0, 0.0, 0.0]),
                           spawn_mass * 100.0, spawn_radius * 2.0, 0.0, sim, kind=2)

        with gui.sub_window("Property Inspector", 0.75, 0.36, 0.24, 0.34) as w:
            sim.selected = w.slider_int("Selected Body Idx", sim.selected, 0, max(sim.num_bodies - 1, 0))
            if sim.num_bodies > 0 and body_active[sim.selected] == 1:
                nm = sim.body_names[sim.selected] if sim.selected < len(sim.body_names) else ""
                if nm:
                    w.text(f"Name: {nm}")
                mp = body_pos[sim.selected]
                mv = body_vel[sim.selected]
                w.text(f"Pos: ({mp.x:.1f}, {mp.y:.1f}, {mp.z:.1f})")
                w.text(f"Vel: ({mv.x:.2f}, {mv.y:.2f}, {mv.z:.2f})")
                new_mass = w.slider_float("Edit Mass", body_mass[sim.selected], 0.01, 30000.0)
                new_rad = w.slider_float("Edit Radius", body_radius[sim.selected], 0.05, 25.0)
                new_temp = w.slider_float("Edit Temp (K)", body_temp[sim.selected], 0.0, 35000.0)
                body_mass[sim.selected] = new_mass
                body_radius[sim.selected] = new_rad
                body_temp[sim.selected] = new_temp
                if w.button("Lock Camera On Body"):
                    sim.follow_body = sim.selected
                    sim.follow_com = False
                if w.button("Unlock Camera"):
                    sim.follow_body = -1
            sim.follow_com = w.checkbox("Track Center-of-Mass", sim.follow_com)

        with gui.sub_window("Relativity Readout", 0.75, 0.72, 0.24, 0.22) as w:
            if sim.num_bodies > 0:
                np_pos = body_pos.to_numpy()[:sim.num_bodies]
                np_mass = body_mass.to_numpy()[:sim.num_bodies]
                np_active = body_active.to_numpy()[:sim.num_bodies]
                factor, src = gravitational_time_dilation(sim.selected, np_pos, np_mass, np_active)
                w.text(f"Time dilation factor: {factor:.6f}")
                if src is not None:
                    w.text(f"Dominant source: body #{src}")
                w.text("(local clock runs at 'factor' x)")
                w.text("relative to a distant observer)")
            else:
                w.text("No bodies in simulation.")

        # ================= MOUSE DRAG-TO-LAUNCH SPAWN TOOL =================
        if paused:
            lmb_down = window.is_pressed(ti.ui.LMB)
            cx, cy = window.get_cursor_pos()
            if lmb_down and not drag_active:
                drag_active = True
                drag_start_pos = (cx, cy)
                fwd = (camera.curr_lookat - camera.curr_position)
                fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
                drag_spawn_point = np.array(camera.curr_position) + fwd * 60.0
            elif lmb_down and drag_active:
                pass
            elif (not lmb_down) and drag_active:
                dx = cx - drag_start_pos[0]
                dy = cy - drag_start_pos[1]
                fwd = (camera.curr_lookat - camera.curr_position)
                fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
                world_up = np.array([0.0, 1.0, 0.0])
                right = np.cross(fwd, world_up)
                right = right / (np.linalg.norm(right) + 1e-6)
                up = np.cross(right, fwd)
                launch_vel = (right * dx + up * dy) * 220.0
                spawn_body(drag_spawn_point, launch_vel, spawn_mass, spawn_radius, spawn_temp, sim)
                drag_active = False

        # ================= PHYSICS STEP =================
        eff_dt = 0.0
        if (not paused) or frame_step_requested:
            direction = -1.0 if reverse_time else 1.0
            eff_dt = base_dt * time_scale * direction
            steps = substeps if not frame_step_requested else 1
            compute_accelerations(sim.num_bodies, epsilon)
            for _ in range(steps):
                verlet_step1(sim.num_bodies, eff_dt)
                compute_accelerations(sim.num_bodies, epsilon)
                verlet_step2(sim.num_bodies, eff_dt)
            process_tidal_disruption(sim)
            frame_step_requested = False
            refresh_body_colors(sim.num_bodies)

        if show_spacetime_grid:
            cam_p_grid = camera.curr_position
            compute_spacetime_grid(sim.num_bodies, cam_p_grid[0], cam_p_grid[2],
                                    grid_extent, grid_depth_scale, 3.0)

        # ================= GRAVITATIONAL LENSING =================
        active_lens_bh = -1
        if lensing_enabled and sim.bh_candidates:
            cam_p = np.array(camera.curr_position)
            best_d = None
            for cand in sim.bh_candidates:
                if cand < sim.num_bodies and body_active[cand] == 1:
                    bp = body_pos[cand]
                    d = (bp.x - cam_p[0]) ** 2 + (bp.y - cam_p[1]) ** 2 + (bp.z - cam_p[2]) ** 2
                    if best_d is None or d < best_d:
                        best_d = d
                        active_lens_bh = cand

        if active_lens_bh >= 0:
            bp = body_pos[active_lens_bh]
            bm = body_mass[active_lens_bh]
            br = body_radius[active_lens_bh]
            bh_rs = max(schwarzschild_radius(bm), br * 1.4) * lens_strength
            cam_p = camera.curr_position
            compute_body_lensing(sim.num_bodies, cam_p[0], cam_p[1], cam_p[2], bp.x, bp.y, bp.z, bh_rs)
            if show_stars:
                compute_star_lensing(cam_p[0], cam_p[1], cam_p[2], bp.x, bp.y, bp.z, bh_rs)
            if show_milkyway:
                compute_mw_lensing(cam_p[0], cam_p[1], cam_p[2], bp.x, bp.y, bp.z, bh_rs)

        # ================= ORBIT TRAILS =================
        if show_trails:
            k = min(sim.num_bodies, TRAIL_MAX_BODIES)
            if k > 0:
                sample = body_pos.to_numpy()[:k]
                _update_trail_lines(_body_trail_buf, sample, k)
                body_trail_lines.from_numpy(_trail_buf_to_lines(_body_trail_buf))

        saturn_visible = (sim.saturn_idx >= 0 and sim.saturn_idx < sim.num_bodies
                          and body_active[sim.saturn_idx] == 1)
        if saturn_visible:
            sp = body_pos[sim.saturn_idx]
            ring_world = _sat_ring_local + np.array([sp.x, sp.y, sp.z], dtype=np.float32)
            saturn_ring_pos.from_numpy(ring_world.astype(np.float32))

        sun_visible = (sim.sol_idx >= 0 and sim.sol_idx < sim.num_bodies
                       and body_active[sim.sol_idx] == 1)
        if sun_visible:
            solp = body_pos[sim.sol_idx]
            sun_c = np.array([solp.x, solp.y, solp.z], dtype=np.float32)
            cam_p = np.array(camera.curr_position, dtype=np.float32)
            axis = cam_p - sun_c
            axis /= (np.linalg.norm(axis) + 1e-6)
            # Corona points that sit between the camera and the sun and project
            # inside its silhouette would render as dark speckles over the white
            # disc (GGUI particles are opaque, they can't add light). Collapse
            # those to the sun's center so its own sphere swallows them, leaving
            # only the halo ring around the limb.
            axial = _sun_glow_local @ axis
            perp = np.linalg.norm(_sun_glow_local - axial[:, None] * axis, axis=1)
            hidden = (axial > 0.0) & (perp < body_radius[sim.sol_idx] * 1.05)
            local = np.where(hidden[:, None], 0.0, _sun_glow_local)
            sun_glow_pos.from_numpy((local + sun_c).astype(np.float32))

        # ================= RENDER =================
        if show_stars:
            n_stars = max(1, int(STAR_COUNT * star_quality))
            star_field_to_draw = render_star_pos if active_lens_bh >= 0 else star_pos
            scene.particles(star_field_to_draw, radius=1.6, per_vertex_color=star_color,
                             index_count=n_stars)
        if show_milkyway:
            n_mw = max(1, int(MILKYWAY_COUNT * star_quality))
            mw_field_to_draw = render_mw_pos if active_lens_bh >= 0 else mw_pos
            scene.particles(mw_field_to_draw, radius=2.2, per_vertex_color=mw_color,
                             index_count=n_mw)
        if show_constellations and n_lines_const > 0:
            scene.lines(const_vertices, width=1.5, color=(0.55, 0.75, 1.0),
                        vertex_count=n_lines_const * 2)
        if show_nebula:
            n_neb = max(1, int(NEBULA_COUNT * nebula_quality))
            scene.particles(nebula_pos, radius=1.1, per_vertex_color=nebula_color,
                             index_count=n_neb)
        if show_bubbles:
            n_bub = max(1, int(BUBBLE_COUNT * nebula_quality))
            scene.particles(bubble_pos, radius=1.4, per_vertex_color=bubble_color,
                             index_count=n_bub)
        if show_globular:
            n_glob = max(1, int(GLOBULAR_COUNT * nebula_quality))
            scene.particles(glob_pos, radius=0.6, per_vertex_color=glob_color,
                             index_count=n_glob)
        if show_spacetime_grid:
            scene.lines(grid_vertices, width=1.0, color=(0.25, 0.55, 0.95))

        if sim.num_bodies > 0:
            body_field_to_draw = render_body_pos if active_lens_bh >= 0 else body_pos
            scene.particles(body_field_to_draw, radius=1.0, per_vertex_color=body_color,
                             index_count=sim.num_bodies, per_vertex_radius=body_radius)

        if saturn_visible:
            scene.particles(saturn_ring_pos, radius=0.09, color=(0.82, 0.76, 0.62),
                             index_count=SATURN_RING_POINTS)

        if sun_visible:
            scene.particles(sun_glow_pos, radius=0.3, per_vertex_color=sun_glow_color,
                             index_count=SUN_GLOW_POINTS, per_vertex_radius=sun_glow_radius)

        if show_trails:
            k = min(sim.num_bodies, TRAIL_MAX_BODIES)
            if k > 0:
                scene.lines(body_trail_lines, width=1.0, color=(0.5, 0.75, 1.0),
                            vertex_count=k * TRAIL_VTX_PER_OBJ)

        canvas.scene(scene)
        window.show()

        # ---------------- FRAME RATE LIMITER ----------------
        last_frame_time = frame_start
        if not fps_unlocked:
            elapsed = time.perf_counter() - frame_start
            budget = 1.0 / target_fps
            if elapsed < budget:
                time.sleep(budget - elapsed)


# ==============================================================================
# 15. SPAWN HELPER
# ==============================================================================
def spawn_body(pos_xyz, vel_xyz, mass, radius, temp, sim: SimState, kind: int = 0):
    if sim.num_bodies >= MAX_BODIES:
        return
    idx = sim.num_bodies
    body_pos[idx] = ti.Vector(list(map(float, pos_xyz)))
    body_vel[idx] = ti.Vector(list(map(float, vel_xyz)))
    body_mass[idx] = float(mass)
    body_radius[idx] = float(radius)
    body_temp[idx] = float(temp)
    body_kind[idx] = int(kind)
    body_active[idx] = 1
    set_single_body_color(idx, float(temp))
    if idx < len(sim.body_names):
        sim.body_names[idx] = ""
    else:
        sim.body_names.append("")
    if kind == 2:
        sim.bh_candidates.append(idx)
    sim.num_bodies += 1


if __name__ == "__main__":
    main()
