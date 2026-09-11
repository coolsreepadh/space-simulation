"""
================================================================================
 GARGANTUA GL -- Taichi N-body physics + a real ModernGL/GLSL ray-marched
 black hole shader, with a minimal imgui simulation-picker panel.
================================================================================
Physics (thousands of particles) runs on the GPU via Taichi kernels -- plain
leapfrog N-body gravity, no Taichi window/GUI involved at all. Every frame the
resulting positions are handed to a separate OpenGL context (pygame creates
the window, ModernGL wraps its GL context) as plain numpy arrays.

Two independent GLSL render paths share that one context:
  - presets 1-3: a vertex/fragment point-sprite shader draws the N-body
    particles directly (plus a static background starfield).
  - preset 4 (Gargantua): a full-screen fragment shader ray-marches every
    pixel's camera ray, bending it step-by-step toward the black hole
    (Delta-dir ~ r_s / b^2), tests the bent ray against the event horizon
    and a thin equatorial accretion disk (blackbody temperature gradient +
    relativistic Doppler beaming), and writes the pixel directly -- no
    particles are drawn for this preset, the disk is 100% analytic.

There is no window frame (borderless) and nothing is drawn over the canvas:
it launches strictly zero-UI. Tab optionally summons one small imgui panel
for picking a simulation by click; everything it offers is on the keyboard
anyway, so it can stay hidden for good.

Controls:
  1-4        switch simulation      space   pause/resume
  right-drag look around            wasd    move            q/e  down/up
  [ / ]      time scale             tab     show/hide picker panel
  esc        quit

Run with:  python gargantua_gl.py
Requires:  pip install taichi numpy moderngl pygame PyOpenGL imgui-bundle
"""

import math
import time

import numpy as np
import taichi as ti
import moderngl
import pygame

from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer

ti.init(arch=ti.gpu, default_fp=ti.f32)

# ==============================================================================
# 1. CONSTANTS
# ==============================================================================
G_CONST = 1.0
C_LIGHT = 60.0
MAX_BODIES = 4002
STAR_COUNT = 8000

WIDTH, HEIGHT = 1600, 900

vec3 = ti.math.vec3


def schwarzschild_radius(mass):
    return 2.0 * G_CONST * mass / (C_LIGHT * C_LIGHT)


def au_to_units(au):
    return 36.0 * math.sqrt(au)


def temp_to_rgb_py(t):
    """CPU-side twin of the GLSL temp_to_rgb() below -- used for particle colors."""
    if t <= 0.0:
        return (0.0, 0.0, 0.0)
    tt = max(1000.0, min(t, 40000.0)) / 100.0
    if tt <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(tt) - 161.1195681661
        b = 0.0 if tt <= 19.0 else 138.5177312231 * math.log(tt - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * (tt - 60.0) ** -0.1332047592
        g = 288.1221695283 * (tt - 60.0) ** -0.0755148492
        b = 255.0
    r = max(0.0, min(r, 255.0)) / 255.0
    g = max(0.0, min(g, 255.0)) / 255.0
    b = max(0.0, min(b, 255.0)) / 255.0
    return (r, g, b)


# ==============================================================================
# 2. TAICHI N-BODY PHYSICS (headless -- no ti.ui window at all)
# ==============================================================================
body_pos    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_vel    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_acc    = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_prevacc = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
body_mass   = ti.field(dtype=ti.f32, shape=MAX_BODIES)
body_active = ti.field(dtype=ti.i32, shape=MAX_BODIES)


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


def load_bodies(pos, vel, mass):
    n = len(mass)
    assert n <= MAX_BODIES
    full_pos = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_vel = np.zeros((MAX_BODIES, 3), dtype=np.float32)
    full_mass = np.zeros(MAX_BODIES, dtype=np.float32)
    full_active = np.zeros(MAX_BODIES, dtype=np.int32)
    full_pos[:n] = pos
    full_vel[:n] = vel
    full_mass[:n] = mass
    full_active[:n] = 1
    body_pos.from_numpy(full_pos)
    body_vel.from_numpy(full_vel)
    body_mass.from_numpy(full_mass)
    body_active.from_numpy(full_active)
    return n


# ==============================================================================
# 3. SIMULATION PRESETS (pure numpy -- position, velocity, mass, color, radius)
# ==============================================================================
def preset_solar_system():
    bodies = [
        ("Sol",     3000.0, 6.5,  5778, 0.0,   0.00, None),
        ("Mercury", 0.055,  0.45, None, 0.39,  0.35, (0.62, 0.58, 0.55)),
        ("Venus",   0.815,  0.95, None, 0.72,  1.85, (0.92, 0.78, 0.55)),
        ("Earth",   1.0,    1.00, None, 1.00,  2.90, (0.30, 0.55, 0.95)),
        ("Mars",    0.107,  0.70, None, 1.52,  4.20, (0.80, 0.40, 0.25)),
        ("Jupiter", 11.0,   2.40, None, 5.20,  0.70, (0.85, 0.70, 0.50)),
        ("Saturn",  9.0,    2.10, None, 9.58,  2.10, (0.90, 0.82, 0.60)),
        ("Uranus",  4.0,    1.50, None, 19.20, 3.60, (0.60, 0.85, 0.90)),
        ("Neptune", 4.0,    1.40, None, 30.05, 5.10, (0.25, 0.35, 0.90)),
    ]
    pos, vel, mass, rad, col = [], [], [], [], []
    sun_mass = bodies[0][1]
    for (_, m, r, t, au, phase, c) in bodies:
        if au == 0.0:
            pos.append([0, 0, 0]); vel.append([0, 0, 0])
        else:
            orbit_r = au_to_units(au)
            x, z = orbit_r * math.cos(phase), orbit_r * math.sin(phase)
            speed = math.sqrt(G_CONST * sun_mass / orbit_r)
            vx, vz = -speed * math.sin(phase), speed * math.cos(phase)
            pos.append([x, 0, z]); vel.append([vx, 0, vz])
        mass.append(m); rad.append(r)
        col.append(c if c is not None else temp_to_rgb_py(t))
    cam_start = (0.0, 150.0, 520.0)
    return np.array(pos), np.array(vel), np.array(mass), np.array(rad), np.array(col), cam_start


def preset_spacetime_demo():
    pos = [[0, 0, 0]]; vel = [[0, 0, 0]]; mass = [8000.0]; rad = [4.0]
    col = [(1.0, 1.0, 1.0)]
    for i in range(6):
        orbit_r = 40.0 + i * 18.0
        phase = i * 1.05
        x, z = orbit_r * math.cos(phase), orbit_r * math.sin(phase)
        speed = math.sqrt(G_CONST * mass[0] / orbit_r)
        vx, vz = -speed * math.sin(phase), speed * math.cos(phase)
        pos.append([x, 0, z]); vel.append([vx, 0, vz])
        mass.append(0.4); rad.append(0.6)
        col.append(temp_to_rgb_py(6000 + i * 400))
    cam_start = (0.0, 60.0, 220.0)
    return np.array(pos), np.array(vel), np.array(mass), np.array(rad), np.array(col), cam_start


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
    return tuple(c * (0.6 + 0.4 * np.random.uniform()))


def preset_galaxy_merger():
    def spiral_disk(center, com_vel, core_mass, n_stars, radius_max, spin_sign, n_arms=2):
        p, v, m, r, c = [], [], [], [], []
        p.append(list(center)); v.append(list(com_vel))
        m.append(core_mass); r.append(0.6); c.append((1.0, 0.95, 0.98))
        for i in range(n_stars):
            rr = radius_max * (0.08 + 0.92 * np.random.uniform() ** 0.5)
            arm_phase = 3.0 * math.log(rr + 1.0)
            arm_offset = (2.0 * math.pi / n_arms) * np.random.randint(0, n_arms)
            ang = arm_phase + arm_offset + np.random.normal(0, 0.3)
            x, z = rr * math.cos(ang), rr * math.sin(ang)
            y = np.random.normal(0, 1.2)
            speed = spin_sign * math.sqrt(G_CONST * core_mass / max(rr, 1.0)) * 0.9
            vx, vz = -speed * math.sin(ang), speed * math.cos(ang)
            p.append([center[0] + x, center[1] + y, center[2] + z])
            v.append([com_vel[0] + vx, com_vel[1], com_vel[2] + vz])
            m.append(0.03); r.append(0.25)
            c.append(_gas_palette_color(rr / radius_max))
        return p, v, m, r, c

    n_each = (MAX_BODIES - 2) // 2
    p1, v1, m1, r1, c1 = spiral_disk((-90, 0, 0), (2.2, 0, 0.4), 1400.0, n_each, 70.0, 1.0)
    p2, v2, m2, r2, c2 = spiral_disk((90, 10, 0), (-2.2, 0, -0.4), 1100.0, n_each, 60.0, -1.0)
    cam_start = (0.0, 90.0, 340.0)
    return (np.array(p1 + p2), np.array(v1 + v2), np.array(m1 + m2),
            np.array(r1 + r2), np.array(c1 + c2), cam_start)


def preset_gargantua():
    """Just the black hole + a doomed star -- the accretion disk itself is
    100% analytic, drawn by the ray-march shader, not by any particle here."""
    pos = [[0.0, 0.0, 0.0]]
    vel = [[0.0, 0.0, 0.0]]
    mass = [3000.0]

    star_orbit_r = 55.0
    star_phase = 0.9
    speed = math.sqrt(G_CONST * mass[0] / star_orbit_r) * 0.82
    sx, sz = star_orbit_r * math.cos(star_phase), star_orbit_r * math.sin(star_phase)
    svx, svz = -speed * math.sin(star_phase), speed * math.cos(star_phase)
    pos.append([sx, 3.0, sz])
    vel.append([svx, 0.0, svz])
    mass.append(140.0)

    cam_start = (0.0, 40.0, 140.0)
    return np.array(pos), np.array(vel), np.array(mass), cam_start


PRESETS = {
    1: ("Solar System", preset_solar_system),
    2: ("Spacetime Curvature Demo", preset_spacetime_demo),
    3: ("Galaxy Merger (Gas Look)", preset_galaxy_merger),
    4: ("Gargantua Black Hole", preset_gargantua),
}

TIDAL_SHED_PER_FRAME = 0.9
STAR_MIN_MASS = 4.0


def process_tidal_feed(n_bodies):
    """Gargantua-preset only: the doomed star (index 1) sheds mass straight
    into the black hole (index 0) once inside its tidal radius -- the hole's
    mass (and so its horizon/disk) visibly grows as the star is consumed."""
    if n_bodies < 2 or body_active[1] == 0:
        return
    bp, sp = body_pos[0], body_pos[1]
    bm, sm = body_mass[0], body_mass[1]
    dx, dy, dz = sp.x - bp.x, sp.y - bp.y, sp.z - bp.z
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r < 1e-4 or sm <= STAR_MIN_MASS:
        return
    tidal_r = 3.2 * (bm / max(sm, 1e-3)) ** (1.0 / 3.0) * 4.0
    if r >= tidal_r:
        return
    shed = min(TIDAL_SHED_PER_FRAME, sm - STAR_MIN_MASS)
    if shed <= 0.0:
        return
    body_mass[1] = sm - shed
    body_mass[0] = bm + shed
    if sm - shed <= STAR_MIN_MASS:
        body_active[1] = 0


# ==============================================================================
# 4. GLSL SHADERS
# ==============================================================================
PARTICLE_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_color;
in float in_radius;
uniform mat4 view;
uniform mat4 proj;
uniform float point_scale;
out vec3 v_color;
void main() {
    vec4 view_pos = view * vec4(in_pos, 1.0);
    gl_Position = proj * view_pos;
    float dist = max(length(view_pos.xyz), 0.001);
    gl_PointSize = clamp(point_scale * in_radius / dist, 2.0, 400.0);
    v_color = in_color;
}
"""

PARTICLE_FRAG = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    float alpha = smoothstep(0.5, 0.15, d);
    frag_color = vec4(v_color * alpha, alpha);
}
"""

FULLSCREEN_VERT = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

GARGANTUA_FRAG = """
#version 330
uniform vec2 resolution;
uniform vec3 cam_pos;
uniform vec3 cam_fwd;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform float fov_scale;
uniform float aspect;
uniform vec3 bh_pos;
uniform float bh_mass;
uniform float bh_rs;
uniform float disk_inner_mult;
uniform float disk_outer_mult;
uniform float lens_strength;
uniform float doppler_boost;
uniform int n_steps;

out vec4 frag_color;

vec3 temp_to_rgb(float t) {
    vec3 res = vec3(0.0);
    if (t > 0.0) {
        float tt = max(1000.0, min(t, 40000.0)) / 100.0;
        float r, g, b;
        if (tt <= 66.0) {
            r = 255.0;
            g = 99.4708025861 * log(tt) - 161.1195681661;
            b = (tt <= 19.0) ? 0.0 : (138.5177312231 * log(tt - 10.0) - 305.0447927307);
        } else {
            r = 329.698727446 * pow(tt - 60.0, -0.1332047592);
            g = 288.1221695283 * pow(tt - 60.0, -0.0755148492);
            b = 255.0;
        }
        r = clamp(r, 0.0, 255.0) / 255.0;
        g = clamp(g, 0.0, 255.0) / 255.0;
        b = clamp(b, 0.0, 255.0) / 255.0;
        res = vec3(r, g, b);
    }
    return res;
}

void main() {
    vec2 ndc = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    float u = ndc.x * aspect;
    float v = ndc.y;
    vec3 dir = normalize(cam_fwd + u * fov_scale * cam_right + v * fov_scale * cam_up);
    vec3 pos = cam_pos;
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float t_total = 0.0;
    const float scene_radius = 320.0;
    const float g_const = 1.0;
    const float c_light = 60.0;

    for (int i = 0; i < n_steps; i++) {
        if (alpha > 0.98 || t_total > scene_radius) break;

        vec3 to_bh = bh_pos - pos;
        float dist = length(to_bh);
        float step_size = 6.0;
        if (dist < 45.0) {
            step_size = 0.5 + 5.5 * clamp(dist / 45.0, 0.0, 1.0);
        }
        if (dist > 1e-4) {
            vec3 bend_dir = to_bh / dist;
            float strength = lens_strength * 1.6 * bh_rs / (dist * dist + 0.15 * bh_rs * bh_rs);
            dir = normalize(dir + bend_dir * strength * step_size);
        }

        vec3 new_pos = pos + dir * step_size;

        if (length(new_pos - bh_pos) < bh_rs) {
            alpha = 1.0;
        }

        if (alpha < 0.98) {
            float y0 = pos.y - bh_pos.y;
            float y1 = new_pos.y - bh_pos.y;
            if (y0 * y1 < 0.0) {
                float tfrac = y0 / (y0 - y1);
                vec3 hit = pos + (new_pos - pos) * tfrac;
                vec3 rel = hit - bh_pos;
                float r_xz = length(vec2(rel.x, rel.z));
                float inner_r = bh_rs * disk_inner_mult;
                float outer_r = bh_rs * disk_outer_mult;
                if (r_xz >= inner_r && r_xz <= outer_r) {
                    float temp = min(30000.0 * pow(inner_r / r_xz, 0.75), 39000.0);
                    vec3 base_col = temp_to_rgb(temp);

                    vec3 tangent = normalize(vec3(-rel.z, 0.0, rel.x));
                    float v_orbit = sqrt(g_const * bh_mass / max(r_xz, bh_rs));
                    float beta = min(v_orbit / c_light, 0.995);
                    float gm = 1.0 / sqrt(1.0 - beta * beta);
                    float cos_th = dot(tangent, -dir);
                    float dop = 1.0 / (gm * (1.0 - beta * cos_th));
                    dop = clamp(dop, 0.05, 6.0);
                    float boost = min(pow(dop, 3.0) * doppler_boost, 14.0);

                    color += (1.0 - alpha) * base_col * boost;
                    alpha = 1.0;
                }
            }
        }

        pos = new_pos;
        t_total += step_size;
    }

    frag_color = vec4(min(color, vec3(1.0)), 1.0);
}
"""


# ==============================================================================
# 5. MATRIX HELPERS (row-major construction, transposed to column-major on upload)
# ==============================================================================
def perspective_matrix(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float64)


def look_at_matrix(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def write_mat4(program, name, m):
    program[name].write(m.T.astype("f4").tobytes())


# ==============================================================================
# 6. FREE CAMERA
# ==============================================================================
class FreeCamera:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.float64)
        to_origin = -self.pos
        self.yaw = math.degrees(math.atan2(to_origin[2], to_origin[0]))
        flat = math.sqrt(to_origin[0] ** 2 + to_origin[2] ** 2)
        self.pitch = math.degrees(math.atan2(to_origin[1], max(flat, 1e-6)))

    def set_position(self, pos):
        self.__init__(pos)

    def basis(self):
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        fwd = np.array([math.cos(pitch) * math.cos(yaw),
                         math.sin(pitch),
                         math.cos(pitch) * math.sin(yaw)])
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        right /= (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, fwd)
        return fwd, right, up

    def update(self, dt, keys, mouse_dx, mouse_dy, looking, move_speed):
        if looking:
            self.yaw += mouse_dx * 0.15
            self.pitch = max(-89.0, min(89.0, self.pitch - mouse_dy * 0.15))
        fwd, right, up = self.basis()
        move = np.zeros(3)
        if keys[pygame.K_w]:
            move += fwd
        if keys[pygame.K_s]:
            move -= fwd
        if keys[pygame.K_d]:
            move += right
        if keys[pygame.K_a]:
            move -= right
        if keys[pygame.K_e]:
            move += up
        if keys[pygame.K_q]:
            move -= up
        n = np.linalg.norm(move)
        if n > 1e-6:
            self.pos += move / n * move_speed * dt


# ==============================================================================
# 7. MAIN APPLICATION
# ==============================================================================
def make_starfield(count, radius):
    pts = np.random.normal(size=(count, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius * (0.85 + 0.15 * np.random.uniform(size=(count, 1)))
    temps = 3000.0 + np.random.uniform(size=count) * 20000.0
    colors = np.array([temp_to_rgb_py(t) for t in temps]) * (0.5 + 0.5 * np.random.uniform(size=(count, 1)))
    radii = np.full(count, 1.2)
    return pts.astype(np.float32), colors.astype(np.float32), radii.astype(np.float32)


def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME
    )
    pygame.display.set_caption("Gargantua GL")
    clock = pygame.time.Clock()

    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    imgui.create_context()
    imgui_impl = PygameRenderer()
    imgui.get_io().display_size = (WIDTH, HEIGHT)

    particle_prog = ctx.program(vertex_shader=PARTICLE_VERT, fragment_shader=PARTICLE_FRAG)
    particle_vbo = ctx.buffer(reserve=(MAX_BODIES + STAR_COUNT) * 7 * 4)
    particle_vao = ctx.vertex_array(
        particle_prog, [(particle_vbo, "3f 3f 1f", "in_pos", "in_color", "in_radius")]
    )

    gargantua_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=GARGANTUA_FRAG)
    quad = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
    quad_vbo = ctx.buffer(quad.tobytes())
    quad_vao = ctx.vertex_array(gargantua_prog, [(quad_vbo, "2f", "in_pos")])

    star_pos, star_color, star_rad = make_starfield(STAR_COUNT, 3800.0)

    camera = FreeCamera((0, 60, 220))
    fov_deg = 45.0
    fov_scale = math.tan(math.radians(fov_deg) / 2.0)
    aspect = WIDTH / HEIGHT

    state = {
        "preset": 0,
        "n_bodies": 0,
        "colors": None,
        "radii": None,
        "paused": False,
        "time_scale": 1.0,
        "epsilon": 1.0,
        "lens_strength": 1.0,
        "doppler_boost": 1.0,
        # Hidden by default: the canvas is strictly zero-UI on launch. Tab
        # summons the small simulation picker if you want to click instead of
        # using the 1-4 keys.
        "show_panel": False,
    }

    def load_preset(idx):
        name, func = PRESETS[idx]
        if idx == 4:
            pos, vel, mass, cam_start = func()
            n = load_bodies(pos, vel, mass)
            state["colors"] = None
            state["radii"] = None
            state["epsilon"] = 3.0
        else:
            pos, vel, mass, rad, col, cam_start = func()
            n = load_bodies(pos, vel, mass)
            state["colors"] = col.astype(np.float32)
            state["radii"] = rad.astype(np.float32)
            state["epsilon"] = 1.0 if idx != 3 else 4.0
        state["preset"] = idx
        state["n_bodies"] = n
        camera.set_position(cam_start)
        state["paused"] = False

    load_preset(1)

    pygame.mouse.set_visible(True)
    running = True
    last_time = time.perf_counter()

    while running:
        now = time.perf_counter()
        dt_wall = now - last_time
        last_time = now

        mouse_dx, mouse_dy = 0, 0
        for event in pygame.event.get():
            imgui_impl.process_event(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    state["paused"] = not state["paused"]
                elif event.key == pygame.K_TAB:
                    state["show_panel"] = not state["show_panel"]
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    load_preset(event.key - pygame.K_0)
            elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[2]:
                mouse_dx, mouse_dy = event.rel

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHTBRACKET]:
            state["time_scale"] = min(state["time_scale"] * 1.03, 1000.0)
        if keys[pygame.K_LEFTBRACKET]:
            state["time_scale"] = max(state["time_scale"] / 1.03, 0.001)

        looking = pygame.mouse.get_pressed()[2]
        camera.update(dt_wall, keys, mouse_dx, mouse_dy, looking, move_speed=80.0)

        # ---------------- PHYSICS ----------------
        n = state["n_bodies"]
        if not state["paused"]:
            dt = 0.02 * state["time_scale"]
            eps = state["epsilon"]
            compute_accelerations(n, eps)
            for _ in range(2):
                verlet_step1(n, dt)
                compute_accelerations(n, eps)
                verlet_step2(n, dt)
            if state["preset"] == 4:
                process_tidal_feed(n)

        # ---------------- IMGUI FRAME ----------------
        # new_frame/render always run so the backend stays in a valid state;
        # when the panel is hidden the draw data is simply empty, so nothing
        # is composited over the canvas.
        imgui_impl.process_inputs()
        imgui.new_frame()
        if state["show_panel"]:
            imgui.set_next_window_size((230, 150), imgui.Cond_.once)
            imgui.set_next_window_pos((12, 12), imgui.Cond_.once)
            imgui.begin("Simulation")
            for idx in (1, 2, 3, 4):
                label, _ = PRESETS[idx]
                selected = state["preset"] == idx
                if selected:
                    imgui.push_style_color(imgui.Col_.button, (0.25, 0.45, 0.85, 1.0))
                if imgui.button(f"{idx}) {label}", (206, 0)):
                    load_preset(idx)
                if selected:
                    imgui.pop_style_color()
            imgui.separator()
            if imgui.button("Pause" if not state["paused"] else "Resume", (206, 0)):
                state["paused"] = not state["paused"]
            imgui.end()

        # ---------------- RENDER ----------------
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0)

        fwd, right, up = camera.basis()

        if state["preset"] == 4:
            bh_pos_v = body_pos[0]
            bh_mass_v = body_mass[0]
            bh_rs = schwarzschild_radius(bh_mass_v)
            gargantua_prog["resolution"].value = (float(WIDTH), float(HEIGHT))
            gargantua_prog["cam_pos"].value = tuple(camera.pos)
            gargantua_prog["cam_fwd"].value = tuple(fwd)
            gargantua_prog["cam_right"].value = tuple(right)
            gargantua_prog["cam_up"].value = tuple(up)
            gargantua_prog["fov_scale"].value = fov_scale
            gargantua_prog["aspect"].value = aspect
            gargantua_prog["bh_pos"].value = (bh_pos_v.x, bh_pos_v.y, bh_pos_v.z)
            gargantua_prog["bh_mass"].value = bh_mass_v
            gargantua_prog["bh_rs"].value = bh_rs
            gargantua_prog["disk_inner_mult"].value = 3.0
            gargantua_prog["disk_outer_mult"].value = 45.0
            gargantua_prog["lens_strength"].value = state["lens_strength"]
            gargantua_prog["doppler_boost"].value = state["doppler_boost"]
            gargantua_prog["n_steps"].value = 90
            quad_vao.render(moderngl.TRIANGLES, vertices=3)
        else:
            view = look_at_matrix(camera.pos, camera.pos + fwd, np.array([0.0, 1.0, 0.0]))
            proj = perspective_matrix(fov_deg, aspect, 0.1, 6000.0)
            write_mat4(particle_prog, "view", view)
            write_mat4(particle_prog, "proj", proj)
            particle_prog["point_scale"].value = 3500.0

            body_np = body_pos.to_numpy()[:n]
            colors = state["colors"]
            radii = state["radii"]

            all_pos = np.concatenate([star_pos, body_np.astype(np.float32)])
            all_col = np.concatenate([star_color, colors])
            all_rad = np.concatenate([star_rad, radii])
            data = np.concatenate(
                [all_pos, all_col, all_rad.reshape(-1, 1)], axis=1
            ).astype(np.float32)
            particle_vbo.write(data.tobytes())
            particle_vao.render(moderngl.POINTS, vertices=len(data))

        imgui.render()
        imgui_impl.render(imgui.get_draw_data())

        pygame.display.flip()
        clock.tick(0)

    imgui_impl.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
