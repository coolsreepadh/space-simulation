"""
================================================================================
 GARGANTUA RAY-MARCHER -- volumetric black hole + N-body galaxy merger
================================================================================
Real-time GPU ray-marcher (written as Taichi kernels -- the whole file is one
Taichi program, so the per-pixel "shader" below is a JIT-compiled CUDA/Vulkan
kernel rather than a separate ModernGL/GLSL program). Each pixel's camera ray
is bent step-by-step toward whichever black hole is nearest (an iterative
weak/strong-field deflection approximation, not a full Kerr geodesic RK4
integrator), tested against a thin equatorial accretion disk (blackbody
temperature profile + relativistic Doppler beaming) and the event horizon,
and blended with a volumetric density field built from the live N-body star
particles -- so the merger's gas/stars are lit as a continuous glow instead
of drawn as individual dots.

Each galaxy's core is a single unified object (mass + horizon + disk): there
is no separate "AGN" entity layered on top of the black hole.

galpy.potential.KeplerPotential.vcirc() sets each star's initial circular
orbital speed around its own core (galpy's internal unit system already
uses G=1, matching this sim's units exactly). The two cores' initial
approach trajectory is a bound, eccentric two-body (Kepler) orbit started at
apoapsis, so the galaxies fall together, swing through a close periapsis
passage (forming tidal tails), and -- via the N-body drag of their own star
disks -- eventually spiral in and merge into a single hole.

Run with:  python gargantua_raymarch.py
Requires:  pip install taichi numpy galpy astropy
================================================================================
"""

import math
import time

import numpy as np
import taichi as ti
from galpy.potential import KeplerPotential

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ==============================================================================
# 1. CONSTANTS
# ==============================================================================
G_CONST = 1.0
C_LIGHT = 60.0

MAX_BODIES = 4002          # index 0, 1 = the two galaxy cores; rest = stars/gas
GRID_N = 64                 # volumetric density grid resolution per axis
GRID_EXTENT = 260.0          # half-width of the density grid (world units)

W, H = 720, 405              # ray-marched image resolution
SCENE_RADIUS = 320.0         # rays beyond this from the camera are "empty space"
DEFAULT_STEPS = 70

vec3 = ti.math.vec3
vec4 = ti.math.vec4

# ==============================================================================
# 2. N-BODY FIELDS (bodies 0/1 are the two unified black-hole+disk cores)
# ==============================================================================
body_pos     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_vel     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_acc     = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_prevacc = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_mass    = ti.field(dtype=ti.f32, shape=MAX_BODIES)
body_active  = ti.field(dtype=ti.i32, shape=MAX_BODIES)
body_kind    = ti.field(dtype=ti.i32, shape=MAX_BODIES)   # 0 = star/gas, 2 = core
body_color   = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)  # gas tint for stars

# Small per-core mirrors, refreshed from body_* each frame -- keeps the
# ray-march kernel's argument list to plain small fields instead of 2x8 scalars.
core_pos_f    = ti.Vector.field(3, dtype=ti.f32, shape=2)
core_mass_f   = ti.field(dtype=ti.f32, shape=2)
core_rs_f     = ti.field(dtype=ti.f32, shape=2)
core_active_f = ti.field(dtype=ti.i32, shape=2)

# Volumetric density grid for the star/gas merger structure
density_grid  = ti.field(dtype=ti.f32, shape=(GRID_N, GRID_N, GRID_N))
density_color = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_N, GRID_N, GRID_N))

image = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))


def schwarzschild_radius(mass):
    return 2.0 * G_CONST * mass / (C_LIGHT * C_LIGHT)


# ==============================================================================
# 3. BLACKBODY COLOR
# ==============================================================================
@ti.func
def temp_to_rgb(t: ti.f32) -> ti.math.vec3:
    res = vec3(0.0, 0.0, 0.0)
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
        res = vec3(r, g, b)
    return res


# ==============================================================================
# 4. N-BODY PHYSICS (identical structure to a plain leapfrog N-body sim)
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


# ==============================================================================
# 5. VOLUMETRIC DENSITY GRID (stars/gas -> continuous glow, no point-cloud)
# ==============================================================================
@ti.kernel
def reset_grid():
    for i, j, k in density_grid:
        density_grid[i, j, k] = 0.0
        density_color[i, j, k] = vec3(0.0, 0.0, 0.0)


@ti.kernel
def deposit_density(n: ti.i32, cx: ti.f32, cy: ti.f32, cz: ti.f32, cell_size: ti.f32):
    half_n = GRID_N // 2
    for i in range(n):
        if body_active[i] == 1 and body_kind[i] == 0:
            p = body_pos[i]
            gx = int((p.x - cx) / cell_size) + half_n
            gy = int((p.y - cy) / cell_size) + half_n
            gz = int((p.z - cz) / cell_size) + half_n
            if 0 <= gx < GRID_N and 0 <= gy < GRID_N and 0 <= gz < GRID_N:
                density_grid[gx, gy, gz] += body_mass[i]
                density_color[gx, gy, gz] += body_mass[i] * body_color[i]


@ti.func
def sample_density(p: ti.math.vec3, cx: ti.f32, cy: ti.f32, cz: ti.f32, cell_size: ti.f32) -> ti.math.vec4:
    """Trilinear-filtered lookup -- avoids the hard voxel edges a nearest-neighbor
    sample would leave visible as a smooth continuous glow is the whole point."""
    half_n = GRID_N // 2
    fx = (p.x - cx) / cell_size + half_n - 0.5
    fy = (p.y - cy) / cell_size + half_n - 0.5
    fz = (p.z - cz) / cell_size + half_n - 0.5
    ix = int(ti.floor(fx))
    iy = int(ti.floor(fy))
    iz = int(ti.floor(fz))
    tx = fx - ix
    ty = fy - iy
    tz = fz - iz
    result = vec4(0.0, 0.0, 0.0, 0.0)
    if 0 <= ix < GRID_N - 1 and 0 <= iy < GRID_N - 1 and 0 <= iz < GRID_N - 1:
        d = 0.0
        c = vec3(0.0, 0.0, 0.0)
        for dxi in ti.static(range(2)):
            for dyi in ti.static(range(2)):
                for dzi in ti.static(range(2)):
                    wx = tx if dxi == 1 else (1.0 - tx)
                    wy = ty if dyi == 1 else (1.0 - ty)
                    wz = tz if dzi == 1 else (1.0 - tz)
                    weight = wx * wy * wz
                    d += weight * density_grid[ix + dxi, iy + dyi, iz + dzi]
                    c += weight * density_color[ix + dxi, iy + dyi, iz + dzi]
        if d > 1e-6:
            avg_c = c / d
            result = vec4(avg_c.x, avg_c.y, avg_c.z, d)
    return result


# ==============================================================================
# 6. RAY MARCHER -- bent camera rays, thin accretion disk, event horizon,
#    volumetric merger glow. Runs entirely as one GPU kernel per frame.
# ==============================================================================
@ti.kernel
def raymarch(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
             fwd_x: ti.f32, fwd_y: ti.f32, fwd_z: ti.f32,
             right_x: ti.f32, right_y: ti.f32, right_z: ti.f32,
             up_x: ti.f32, up_y: ti.f32, up_z: ti.f32,
             fov_scale: ti.f32, aspect: ti.f32,
             disk_inner_mult: ti.f32, disk_outer_mult: ti.f32,
             doppler_boost: ti.f32, lens_strength: ti.f32,
             grid_cx: ti.f32, grid_cy: ti.f32, grid_cz: ti.f32, cell_size: ti.f32,
             show_volume: ti.i32, n_steps: ti.i32):
    cam = vec3(cam_x, cam_y, cam_z)
    fwd = vec3(fwd_x, fwd_y, fwd_z)
    right = vec3(right_x, right_y, right_z)
    up = vec3(up_x, up_y, up_z)

    for px, py in image:
        u = (2.0 * (px + 0.5) / W - 1.0) * aspect
        v = (1.0 - 2.0 * (py + 0.5) / H)
        d = (fwd + u * fov_scale * right + v * fov_scale * up).normalized()
        pos = cam
        color = vec3(0.0, 0.0, 0.0)
        alpha = 0.0
        t_total = 0.0

        for _step in range(n_steps):
            if alpha > 0.98 or t_total > SCENE_RADIUS:
                break

            min_dist = 1.0e9
            closest = -1
            for c in ti.static(range(2)):
                if core_active_f[c] == 1:
                    dd = (core_pos_f[c] - pos).norm()
                    if dd < min_dist:
                        min_dist = dd
                        closest = c

            step_size = 6.0
            if min_dist < 45.0:
                step_size = 0.5 + 5.5 * ti.min(min_dist / 45.0, 1.0)

            if closest >= 0:
                to_c = core_pos_f[closest] - pos
                dist = to_c.norm()
                if dist > 1e-4:
                    rs = core_rs_f[closest]
                    bend_dir = to_c / dist
                    strength = lens_strength * 1.6 * rs / (dist * dist + 0.15 * rs * rs)
                    d = (d + bend_dir * strength * step_size).normalized()

            new_pos = pos + d * step_size

            for c in ti.static(range(2)):
                if core_active_f[c] == 1 and alpha < 0.98:
                    cpos = core_pos_f[c]
                    rs = core_rs_f[c]

                    # event horizon capture
                    if (new_pos - cpos).norm() < rs:
                        alpha = 1.0

                    # thin equatorial disk crossing
                    if alpha < 0.98:
                        y0 = pos.y - cpos.y
                        y1 = new_pos.y - cpos.y
                        if y0 * y1 < 0.0:
                            tfrac = y0 / (y0 - y1)
                            hit = pos + (new_pos - pos) * tfrac
                            rel = hit - cpos
                            r_xz = ti.sqrt(rel.x * rel.x + rel.z * rel.z)
                            inner_r = rs * disk_inner_mult
                            outer_r = rs * disk_outer_mult
                            if inner_r <= r_xz <= outer_r:
                                temp = 30000.0 * ti.pow(inner_r / r_xz, 0.75)
                                temp = ti.min(temp, 39000.0)
                                base_col = temp_to_rgb(temp)

                                spin_sign = 1.0 if c == 0 else -1.0
                                tangent = vec3(-rel.z, 0.0, rel.x).normalized() * spin_sign
                                v_orbit = ti.sqrt(G_CONST * core_mass_f[c] / ti.max(r_xz, rs))
                                beta = ti.min(v_orbit / C_LIGHT, 0.995)
                                gamma = 1.0 / ti.sqrt(1.0 - beta * beta)
                                cos_th = tangent.dot(-d)
                                dop = 1.0 / (gamma * (1.0 - beta * cos_th))
                                dop = ti.min(ti.max(dop, 0.05), 6.0)
                                boost = ti.min(ti.pow(dop, 3.0) * doppler_boost, 14.0)

                                color += (1.0 - alpha) * base_col * boost
                                alpha = 1.0

            near_a_disk = 0
            if closest >= 0 and min_dist < core_rs_f[closest] * disk_outer_mult * 1.15:
                near_a_disk = 1

            if show_volume == 1 and alpha < 0.98 and near_a_disk == 0:
                samp = sample_density(pos, grid_cx, grid_cy, grid_cz, cell_size)
                dens = samp.w
                if dens > 1e-6:
                    emit = vec3(samp.x, samp.y, samp.z) * ti.min(dens * 0.5, 3.0)
                    contrib = ti.min(dens * 0.03 * step_size, 0.35)
                    color += (1.0 - alpha) * emit * contrib
                    alpha = ti.min(alpha + contrib, 1.0)

            pos = new_pos
            t_total += step_size

        image[px, py] = ti.min(color, 1.0)


# ==============================================================================
# 7. SCENE SETUP -- galpy for per-galaxy circular orbits, Kepler two-body
#    conic section (started at apoapsis) for the approach trajectory.
# ==============================================================================
def _gas_palette_color(radial_frac):
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
    c = c * (0.6 + 0.4 * np.random.uniform())
    return c


def build_merger_scene(core_mass_a=3000.0, core_mass_b=2400.0,
                        n_stars_each=1999, disk_radius=80.0,
                        peri=27.0, ecc=0.75):
    """Two galaxy disks (galpy-vcirc orbits) on a bound, eccentric two-body
    approach trajectory (started at apoapsis) so they swing through a close
    passage, raise tidal tails, and -- via the N-body drag of their own star
    disks -- eventually spiral together."""
    m_tot = core_mass_a + core_mass_b
    a_semi = peri / (1.0 - ecc)          # semi-major axis of the core-core orbit
    r_apo = a_semi * (1.0 + ecc)
    v_rel_apo = math.sqrt(G_CONST * m_tot / a_semi * (1.0 - ecc) / (1.0 + ecc))

    # split around the center of mass by the (inverse) mass ratio
    r_a = (core_mass_b / m_tot) * r_apo
    r_b = (core_mass_a / m_tot) * r_apo
    v_a = (core_mass_b / m_tot) * v_rel_apo
    v_b = (core_mass_a / m_tot) * v_rel_apo

    core_a_pos = np.array([-r_a, 0.0, 0.0])
    core_b_pos = np.array([r_b, 0.0, 0.0])
    core_a_vel = np.array([0.0, 0.0, -v_a])
    core_b_vel = np.array([0.0, 0.0, v_b])

    pos, vel, mass, active, kind, color = [], [], [], [], [], []
    pos += [core_a_pos.tolist(), core_b_pos.tolist()]
    vel += [core_a_vel.tolist(), core_b_vel.tolist()]
    mass += [core_mass_a, core_mass_b]
    active += [1, 1]
    kind += [2, 2]
    color += [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]  # unused for kind==2 (drawn by the ray marcher)

    for (center, com_vel, core_mass, spin_sign) in [
        (core_a_pos, core_a_vel, core_mass_a, 1.0),
        (core_b_pos, core_b_vel, core_mass_b, -1.0),
    ]:
        kp = KeplerPotential(amp=core_mass)  # galpy natural units: G == 1, matches this sim
        n_arms = 2
        for i in range(n_stars_each):
            rr = disk_radius * (0.08 + 0.92 * np.random.uniform() ** 0.5)
            arm_phase = 3.0 * math.log(rr + 1.0)
            arm_offset = (2.0 * math.pi / n_arms) * np.random.randint(0, n_arms)
            ang = arm_phase + arm_offset + np.random.normal(0, 0.3)
            x = rr * math.cos(ang)
            z = rr * math.sin(ang)
            y = np.random.normal(0, 1.2)
            speed = spin_sign * float(kp.vcirc(max(rr, 1.0))) * 0.92
            vx = -speed * math.sin(ang)
            vz = speed * math.cos(ang)
            pos.append([center[0] + x, center[1] + y, center[2] + z])
            vel.append([com_vel[0] + vx, com_vel[1], com_vel[2] + vz])
            mass.append(0.03)
            active.append(1)
            kind.append(0)
            color.append(_gas_palette_color(rr / disk_radius).tolist())

    return (np.array(pos, dtype=np.float32), np.array(vel, dtype=np.float32),
            np.array(mass, dtype=np.float32), np.array(active, dtype=np.int32),
            np.array(kind, dtype=np.int32), np.array(color, dtype=np.float32))


def load_scene(pos, vel, mass, active, kind, color):
    n = len(mass)
    assert n <= MAX_BODIES
    full_pos = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_vel = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_mass = np.zeros(MAX_BODIES, dtype=np.float32)
    full_active = np.zeros(MAX_BODIES, dtype=np.int32)
    full_kind = np.zeros(MAX_BODIES, dtype=np.int32)
    full_color = np.zeros((MAX_BODIES, 3), dtype=np.float32)

    full_pos[:n] = pos
    full_vel[:n] = vel
    full_mass[:n] = mass
    full_active[:n] = active
    full_kind[:n] = kind
    full_color[:n] = color

    body_pos.from_numpy(full_pos)
    body_vel.from_numpy(full_vel)
    body_mass.from_numpy(full_mass)
    body_active.from_numpy(full_active)
    body_kind.from_numpy(full_kind)
    body_color.from_numpy(full_color)
    return n


def sync_core_fields():
    """Mirror bodies 0/1 into the small core_* fields the ray marcher reads."""
    for c in range(2):
        p = body_pos[c]
        core_pos_f[c] = ti.Vector([p.x, p.y, p.z])
        m = body_mass[c]
        core_mass_f[c] = m
        core_rs_f[c] = schwarzschild_radius(m)
        core_active_f[c] = body_active[c]


CORE_MERGE_FACTOR = 3.0  # merge once separation < this many horizon radii


def process_core_merger():
    if body_active[0] == 1 and body_active[1] == 1:
        p0, p1 = body_pos[0], body_pos[1]
        dx, dy, dz = p0.x - p1.x, p0.y - p1.y, p0.z - p1.z
        sep = math.sqrt(dx * dx + dy * dy + dz * dz)
        m0, m1 = body_mass[0], body_mass[1]
        rs0, rs1 = schwarzschild_radius(m0), schwarzschild_radius(m1)
        if sep < CORE_MERGE_FACTOR * (rs0 + rs1):
            v0, v1 = body_vel[0], body_vel[1]
            new_mass = m0 + m1
            npx = (p0.x * m0 + p1.x * m1) / new_mass
            npy = (p0.y * m0 + p1.y * m1) / new_mass
            npz = (p0.z * m0 + p1.z * m1) / new_mass
            nvx = (v0.x * m0 + v1.x * m1) / new_mass
            nvy = (v0.y * m0 + v1.y * m1) / new_mass
            nvz = (v0.z * m0 + v1.z * m1) / new_mass
            body_mass[0] = new_mass
            body_pos[0] = ti.Vector([npx, npy, npz])
            body_vel[0] = ti.Vector([nvx, nvy, nvz])
            body_active[1] = 0
            return True
    return False


# ==============================================================================
# 8. MAIN APPLICATION
# ==============================================================================
def main():
    pos, vel, mass, active, kind, color = build_merger_scene()
    n_bodies = load_scene(pos, vel, mass, active, kind, color)
    sync_core_fields()

    # Plain render window -- no GUI object is ever created, so there is
    # nothing available to draw a panel, slider, or text overlay with.
    window = ti.ui.Window("Gargantua Ray Marcher", (W, H), vsync=True, show_window=True)
    canvas = window.get_canvas()
    camera = ti.ui.Camera()
    camera.position(0, 55, 230)
    camera.lookat(0, 0, 0)
    camera.up(0, 1, 0)

    paused = False
    time_scale = 1.0
    base_dt = 0.02
    epsilon = 5.0
    substeps = 2
    n_steps = DEFAULT_STEPS
    lens_strength = 1.0
    doppler_boost = 1.0
    show_volume = True
    merged = False
    space_was_down = False

    grid_cell_size = (2.0 * GRID_EXTENT) / GRID_N
    fov_scale = math.tan(math.radians(45.0) * 0.5)

    while window.running:
        camera.track_user_inputs(window, movement_speed=1.2, hold_key=ti.ui.RMB)

        # Silent keyboard controls -- no on-screen indication, nothing drawn:
        #   space = pause/resume   |   [ / ] = time scale   |   - / = = lens strength
        space_down = window.is_pressed(ti.ui.SPACE)
        if space_down and not space_was_down:
            paused = not paused
        space_was_down = space_down
        if window.is_pressed("]"):
            time_scale = min(time_scale * 1.03, 8.0)
        if window.is_pressed("["):
            time_scale = max(time_scale / 1.03, 0.02)
        if window.is_pressed("="):
            lens_strength = min(lens_strength + 0.02, 4.0)
        if window.is_pressed("-"):
            lens_strength = max(lens_strength - 0.02, 0.0)

        # ---------------- PHYSICS ----------------
        if not paused:
            dt = base_dt * time_scale
            compute_accelerations(n_bodies, epsilon)
            for _ in range(substeps):
                verlet_step1(n_bodies, dt)
                compute_accelerations(n_bodies, epsilon)
                verlet_step2(n_bodies, dt)
            if not merged:
                merged = process_core_merger()
            sync_core_fields()

            reset_grid()
            gp = body_pos[0]
            gcom = np.array([gp.x, gp.y, gp.z]) if body_active[0] == 1 else np.zeros(3)
            deposit_density(n_bodies, float(gcom[0]), float(gcom[1]), float(gcom[2]), grid_cell_size)

        # ---------------- RAY MARCH ----------------
        cam_pos = np.array(camera.curr_position)
        fwd = np.array(camera.curr_lookat) - cam_pos
        fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, fwd)
        aspect = W / H
        gp = body_pos[0]
        gcom = np.array([gp.x, gp.y, gp.z]) if body_active[0] == 1 else np.zeros(3)

        raymarch(cam_pos[0], cam_pos[1], cam_pos[2],
                 fwd[0], fwd[1], fwd[2],
                 right[0], right[1], right[2],
                 up[0], up[1], up[2],
                 fov_scale, aspect,
                 3.0, 45.0,
                 doppler_boost, lens_strength,
                 float(gcom[0]), float(gcom[1]), float(gcom[2]), grid_cell_size,
                 1 if show_volume else 0, n_steps)

        canvas.set_image(image)
        window.show()


if __name__ == "__main__":
    main()
