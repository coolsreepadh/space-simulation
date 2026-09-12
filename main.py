"""
================================================================================
 GPU N-BODY ASTROPHYSICS SIMULATOR + RAY-MARCHED GARGANTUA
================================================================================
A Taichi/CUDA N-body integrator feeding a ModernGL render stack whose final pass
ray-marches null geodesics around a Schwarzschild black hole, with a Dear ImGui
control panel over the top.

Run with:  python main.py
Requires:  pip install taichi numpy moderngl pygame PyOpenGL imgui-bundle

Pipeline per frame
------------------
    Taichi (GPU)  ->  positions
    pass 1  particles   -> HDR scene FBO  (additive point sprites)
    pass 2  lensing     -> HDR lens FBO   (ray-marched geodesics that re-sample
                                           the scene FBO along the *bent* ray
                                           direction, so the N-body particles
                                           are themselves gravitationally
                                           lensed), or a flat pass-through for
                                           the non-black-hole scenes
    pass 3  bright-pass + separable blur  (quarter res bloom)
    pass 4  composite: ACES tonemap + vignette + dither -> screen
    pass 5  Dear ImGui panels

Scenes (number keys, or the Scenes panel)
-----------------------------------------
    1  Solar System disk      Sun, all 8 planets + the 5 IAU dwarf planets on
                              their real Keplerian orbits (true eccentricity
                              and inclination, not flattened circles), major
                              moons, Saturn's rings, asteroid/Kuiper belts,
                              and real space-mission trajectories.  The Oort
                              cloud and the heliosphere are switchable in the
                              "Solar System" panel: both are far enough out
                              that seeing either means leaving the planets as
                              a knot in the middle
    2  Galaxy Merger          two live spiral galaxies, bulge + halo + disk,
                              on a grazing prograde encounter -- bridge,
                              tidal tails, merger
    3  Black hole (10 M_sun) a BARE Schwarzschild hole -- press X to drop a
                              star on it and watch tides shred it into a
                              stream that circularises into the disk
    4  Star Cluster Collapse  cold Plummer sphere, full O(N^2) self-gravity

Controls
--------
    1-4          switch scene            R         restart current scene
    LMB drag     orbit                   wheel     zoom
    MMB drag     pan                     shift+LMB pan
    arrow keys   pan                     W/S       dolly in/out
    A/D          orbit left/right        Q/E       orbit down/up
    SPACE        pause                   [ / ]     slower / faster
    L            toggle lensing          F         reset view
    N            toggle body names       O         toggle orbit paths
    X            spawn a star (scene 3)  V         draw material as gas
    G            hide/show panels        F11       fullscreen
    ESC          quit

The legacy Taichi-GGUI build is kept alongside as main_ggui_legacy.py.
================================================================================
"""

import argparse
import math
import time

import numpy as np

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--preset", type=int, default=2, choices=(1, 2, 3, 4),
                    help="scene to start in (default: 2, galaxy merger)")
    ap.add_argument("--fullscreen", action="store_true",
                    help="start fullscreen instead of in a resizable window")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.85,
                    help="internal render scale; lower = faster lensing")
    ap.add_argument("--steps", type=int, default=260,
                    help="max geodesic integration steps per pixel")
    ap.add_argument("--halo", type=int, default=1400,
                    help="live halo particles per galaxy (massive)")
    ap.add_argument("--disk", type=int, default=12000,
                    help="disk particles per galaxy (tracers)")
    ap.add_argument("--arch", default="gpu", choices=("gpu", "cuda", "vulkan", "cpu"))
    ap.add_argument("--frames", type=int, default=0,
                    help="exit after N frames (smoke test)")
    ap.add_argument("--vsync", type=int, default=1)
    ap.add_argument("--shot", default="",
                    help="PNG path prefix; frames listed in --shotframes are saved")
    ap.add_argument("--shotframes", default="",
                    help="comma separated frame numbers to capture")
    return ap.parse_args(argv)


ARGS = parse_args()

import taichi as ti  # noqa: E402  (after argparse so --help stays fast)

_ARCH = {"gpu": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[ARGS.arch]
ti.init(arch=_ARCH, default_fp=ti.f32, random_seed=7, offline_cache=True)

import moderngl  # noqa: E402
import pygame    # noqa: E402
from OpenGL import GL as gl  # noqa: E402  (imgui backend needs it anyway)
from imgui_bundle import imgui  # noqa: E402
from imgui_bundle.python_backends.pygame_backend import PygameRenderer  # noqa: E402


# ---------------------------------------------------------------------------
# Taichi state
# ---------------------------------------------------------------------------

MAX_N = 80000

pos  = ti.Vector.field(3, ti.f32, shape=MAX_N)
vel  = ti.Vector.field(3, ti.f32, shape=MAX_N)
acc  = ti.Vector.field(3, ti.f32, shape=MAX_N)
mass = ti.field(ti.f32, shape=MAX_N)
sft2 = ti.field(ti.f32, shape=MAX_N)   # per-source squared Plummer softening

MIN_SOFT2 = 1e-4


@ti.kernel
def k_upload(n: ti.i32,
             p: ti.types.ndarray(), v: ti.types.ndarray(),
             m: ti.types.ndarray(), s: ti.types.ndarray()):
    for i in range(n):
        pos[i]  = ti.Vector([p[i, 0], p[i, 1], p[i, 2]])
        vel[i]  = ti.Vector([v[i, 0], v[i, 1], v[i, 2]])
        acc[i]  = ti.Vector([0.0, 0.0, 0.0])
        mass[i] = m[i]
        sft2[i] = ti.max(s[i] * s[i], MIN_SOFT2)


@ti.kernel
def k_accel(n: ti.i32, src_lo: ti.i32, src_hi: ti.i32,
            gconst: ti.f32, pw_rs: ti.f32):
    """Direct-sum gravity.

    Sources are indices [src_lo, src_hi): the massive bodies.  Every particle
    in [0, n) is accelerated by them, so a scene can mix live self-gravitating
    matter with massless tracers at a fraction of the O(N^2) cost.

    The source set is a *range* rather than a prefix so that the black hole
    scene can give self-gravity to just the one star that is still whole
    while several earlier debris streams are on the board: those are long
    since dominated by the hole, and including them would push the O(N^2)
    inner loop up by the square of the number of stars for no visible gain.

    If pw_rs > 0 the body at index 0 is the hole and is always a source,
    outside the range, through the Paczynski-Wiita pseudo-Newtonian
    potential -GM/(r - r_s), which reproduces the Schwarzschild ISCO at 3 r_s
    and makes orbits inside it plunge.  Callers in that case pass src_lo >= 1
    so it is not also counted as an ordinary Newtonian source.
    """
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0])
        pi = pos[i]
        if pw_rs > 0.0:
            d = pos[0] - pi
            r2 = d.dot(d) + sft2[0]
            r = ti.sqrt(r2)
            rr = ti.max(r - pw_rs, 0.35 * pw_rs)
            a += (gconst * mass[0] / (rr * rr * r)) * d
        for j in range(src_lo, src_hi):
            d = pos[j] - pi
            r2 = d.dot(d) + sft2[j]
            r = ti.sqrt(r2)
            a += (gconst * mass[j] / (r2 * r)) * d
        acc[i] = a


@ti.kernel
def k_kick(n: ti.i32, dt: ti.f32):
    for i in range(n):
        vel[i] += acc[i] * dt


@ti.kernel
def k_drift(n: ti.i32, dt: ti.f32):
    for i in range(n):
        pos[i] += vel[i] * dt


@ti.kernel
def k_recycle(lo: ti.i32, hi: ti.i32, gm: ti.f32, rs: ti.f32,
              r_in: ti.f32, r_out: ti.f32, thick: ti.f32):
    """Anything that crosses the horizon is re-injected at the outer disk edge,
    so the accretion disk stays fed instead of slowly draining away."""
    for i in range(lo, hi):
        r = pos[i].norm()
        if r < 1.02 * rs or r > 40.0 * r_out:
            ang = ti.random() * 6.2831853
            rr = r_out * (0.82 + 0.18 * ti.random())
            pos[i] = ti.Vector([rr * ti.cos(ang),
                                (ti.random() - 0.5) * thick,
                                rr * ti.sin(ang)])
            vc = ti.sqrt(gm * rr) / ti.max(rr - rs, 1e-3)
            vc *= 0.995 + 0.01 * ti.random()
            vel[i] = ti.Vector([-vc * ti.sin(ang), 0.0, vc * ti.cos(ang)])


# --- physical scale of the hole ---------------------------------------------
#
# The black hole scene runs in geometric units: 1 world unit = 1 Schwarzschild
# radius and c = 1, so GM = r_s / 2 = 0.5.  Schwarzschild dynamics measured in
# r_s are mass-independent -- a hole of any mass looks identical this way -- so
# pinning the mass is purely a question of what the units MEAN.  At 10 solar
# masses one world unit is about 30 km, which is what the readouts convert to.

BH_SOLAR_MASSES = 10.0
_SOLAR_MASS_KG = 1.98892e30
_G_SI = 6.67430e-11
_C_SI = 2.99792458e8
# r_s = 2GM/c^2, in km
BH_RS_KM = (2.0 * _G_SI * BH_SOLAR_MASSES * _SOLAR_MASS_KG / (_C_SI ** 2)) / 1000.0
# the hole is 0.5 in sim mass units, so this converts sim mass -> solar masses
SIM_MASS_TO_SOLAR = BH_SOLAR_MASSES / 0.5


# --- tidal disruption: spawning a star, accreting it, circularising it ------
#
# The black hole scene starts bare.  A star is spawned as a live,
# self-gravitating Plummer ball parked outside the array's active range until
# then; self-gravity is what lets it hold together on the way in and then lose
# to the tide at pericentre, instead of shearing apart from frame one.

GRAVEYARD = 6.0e4      # where accreted particles are parked: far enough that
                       # the point sprite's distance falloff makes them vanish

_swallowed = ti.field(ti.i32, shape=1)
_red = ti.Vector.field(4, ti.f32, shape=1)   # (m*vx, m*vy, m*vz, m) accumulator


@ti.kernel
def k_write_block(lo: ti.i32, count: ti.i32,
                  p: ti.types.ndarray(), v: ti.types.ndarray(),
                  m: ti.types.ndarray(), s: ti.types.ndarray()):
    for k in range(count):
        i = lo + k
        pos[i] = ti.Vector([p[k, 0], p[k, 1], p[k, 2]])
        vel[i] = ti.Vector([v[k, 0], v[k, 1], v[k, 2]])
        acc[i] = ti.Vector([0.0, 0.0, 0.0])
        mass[i] = m[k]
        sft2[i] = ti.max(s[k] * s[k], MIN_SOFT2)


@ti.kernel
def k_swallow(lo: ti.i32, hi: ti.i32, rs: ti.f32, sustain: ti.i32,
              gm: ti.f32, feed_r: ti.f32, r_return: ti.f32,
              jet_range: ti.f32) -> ti.i32:
    """Handle whatever reaches the hole.

    Material put back is given a scale height and a little vertical motion:
    dropped into the midplane exactly, and then damped towards it, the disk
    ends up with no thickness at all, and a disk of no thickness is a hard
    bright line on screen rather than a band of gas.

    With sustain off it is simply gone: massless and parked far away, so the
    disk drains and the hole eventually finishes its meal.  With sustain on the
    same material is resupplied at the feed radius on a circular orbit, which
    is how a real stellar-mass hole keeps a disk at all -- Cygnus X-1 and its
    kin are fed continuously by a companion, and their disks persist rather
    than being a one-off meal that empties out.
    """
    _swallowed[0] = 0
    for i in range(lo, hi):
        # Anything flung far out is brought back rather than written off.  A
        # little material always gets scattered onto an escaping orbit, and
        # without this the disk quietly bleeds particles for the whole run
        # instead of staying put.
        r_i = pos[i].norm()
        limit = r_return
        if abs(pos[i][1]) > 0.45 * r_i:
            limit = jet_range      # jet material loops back sooner, keeping the
                                   # beam short and dense enough to read as one
        # Parked-at-the-graveyard is the test for "already dealt with", not
        # zero mass: most of a star is massless tracer particles, and keying
        # off mass would let all of them fall straight through the horizon.
        live = r_i < 0.5 * GRAVEYARD
        if sustain == 1 and live and r_i > limit:
            ang = ti.random() * 6.2831853
            rr = feed_r * (0.90 + 0.20 * ti.random())
            pos[i] = ti.Vector([rr * ti.cos(ang),
                                (ti.random() - 0.5) * 0.06 * rr,
                                rr * ti.sin(ang)])
            vc = ti.sqrt(gm * rr) / ti.max(rr - rs, 1e-3)
            vel[i] = ti.Vector([-vc * ti.sin(ang),
                                (ti.random() - 0.5) * 0.06 * vc,
                                vc * ti.cos(ang)])
        # 2.5 r_s, not the horizon itself: anything this far inside the 3 r_s
        # ISCO has no stable orbit left and is already committed to falling in,
        # and capturing it out here keeps it well clear of the radius where the
        # Paczynski-Wiita force gets steep enough that a finite timestep would
        # slingshot it straight back out at escape speed.  Checked every
        # substep for the same reason -- one frame of travel is enough for a
        # fast plunging orbit to dive deep between checks.
        if live and pos[i].norm() < 2.5 * rs:
            _swallowed[0] += 1
            if sustain == 1:
                ang = ti.random() * 6.2831853
                rr = feed_r * (0.90 + 0.20 * ti.random())
                pos[i] = ti.Vector([rr * ti.cos(ang),
                                    (ti.random() - 0.5) * 0.06 * rr,
                                    rr * ti.sin(ang)])
                vc = ti.sqrt(gm * rr) / ti.max(rr - rs, 1e-3)
                vel[i] = ti.Vector([-vc * ti.sin(ang),
                                    (ti.random() - 0.5) * 0.06 * vc,
                                    vc * ti.cos(ang)])
            else:
                mass[i] = 0.0
                vel[i] = ti.Vector([0.0, 0.0, 0.0])
                pos[i] = ti.Vector([GRAVEYARD + 3.0 * ti.cast(i % 97, ti.f32),
                                    GRAVEYARD, GRAVEYARD])
    return _swallowed[0]


@ti.kernel
def k_jet(lo: ti.i32, hi: ti.i32, rate: ti.f32, r_source: ti.f32,
          v_jet: ti.f32, spread: ti.f32, base: ti.f32, rs: ti.f32):
    """Launch a trickle of disk material back out along the poles.

    Jetted tidal disruptions are a real if uncommon class -- Swift J1644+57 is
    the famous one -- and the twin beams perpendicular to the disk are the most
    recognisable thing about the artist renderings of them.

    Material is drawn from anywhere in the disk body but always launched from
    the axis just above the hole, which is an idealisation: drawing only from
    the handful of particles that happen to be at the launch radius leaves the
    beam far too sparse to read as a beam, and launching from wherever they
    already were gives a broad fountain rather than a collimated jet.  What it
    costs the disk is returned -- k_swallow brings the material back once it
    coasts past the return radius -- so the jet is a loop, not a leak.
    """
    for i in range(lo, hi):
        r = pos[i].norm()
        if mass[i] >= 0.0 and r > 2.0 * rs and r < r_source and ti.random() < rate:
            sign = 1.0
            if ti.random() < 0.5:
                sign = -1.0
            ang = ti.random() * 6.2831853
            rad = 0.30 * ti.sqrt(ti.random())
            pos[i] = ti.Vector([rad * ti.cos(ang), sign * base, rad * ti.sin(ang)])
            lat = spread * v_jet * ti.sqrt(ti.random())
            vel[i] = ti.Vector([lat * ti.cos(ang), sign * v_jet, lat * ti.sin(ang)])


@ti.kernel
def k_balance_momentum(n: ti.i32):
    """Give the central body the recoil that keeps total momentum at zero.

    With a low-mass star this is a rounding error, but a star heavy enough to
    matter would otherwise hand the whole system a net drift and walk it out
    of frame over a few thousand time units."""
    _red[0] = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in range(1, n):
        mv = mass[i] * vel[i]
        _red[0] += ti.Vector([mv[0], mv[1], mv[2], 0.0])
    if mass[0] > 0.0:
        vel[0] = -ti.Vector([_red[0][0], _red[0][1], _red[0][2]]) / mass[0]


@ti.kernel
def k_recentre(n: ti.i32):
    """Put the system's centre of mass back at the origin.

    The hole's velocity is pinned by k_balance_momentum, but nothing pins its
    position, and several things here do not conserve momentum on their own:
    the viscous damping bleeds it out of the gas, and the swallow-and-resupply
    teleports move material across the box in a single step.  Each of those
    makes the hole's pinned velocity jump, and the integral of those jumps is
    a random walk.  With the default feather-weight star it is invisible; give
    the star a few solar masses and the hole walks clean out of its own disk
    inside a few thousand time units.  Pinning the barycentre costs nothing and
    still lets the hole orbit it, which with a heavy star it genuinely should.
    """
    _red[0] = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in range(n):
        m = mass[i]
        if m > 0.0:
            _red[0] += ti.Vector([m * pos[i][0], m * pos[i][1], m * pos[i][2], m])
    tm = _red[0][3]
    if tm > 0.0:
        c = ti.Vector([_red[0][0], _red[0][1], _red[0][2]]) / tm
        for i in range(n):
            pos[i] -= c


@ti.kernel
def k_circularise(lo: ti.i32, hi: ti.i32, frac: ti.f32, tang: ti.f32,
                  r_max: ti.f32, rs: ti.f32):
    """Stand-in for the viscous dissipation a collisionless N-body cannot have.

    Real tidal debris only settles into a disk because the returning stream
    shocks against itself and radiates the energy away; with no dissipation at
    all the stream just precesses forever into an eccentric fan.  Two separate
    effects, because they do different jobs:

      * damping the radial (and vertical) component at fixed angular momentum
        drives an eccentric orbit onto the circular one carrying the same L --
        this is what turns the stream into a disk;
      * bleeding a much smaller slice off the tangential component removes L,
        which is what makes the disk spread inward and actually drain into the
        hole.  Without it the debris circularises into a dead static ring at
        whatever radius its angular momentum happens to match, and the hole
        never gets to eat.
    """
    for i in range(lo, hi):
        r = pos[i].norm()
        # |y| < 0.3 r keeps this to material that is actually in the disk --
        # jet particles climbing out along the poles must not be dragged back
        # down into the plane by it
        if r < r_max and r > 1.5 * rs and abs(pos[i][1]) < 0.3 * r:
            rhat = pos[i] / r
            vr = vel[i].dot(rhat)
            vel[i] -= rhat * (vr * frac)
            # only lightly: the vertical motion is what gives the disk its
            # thickness, and damping it as hard as the radial component
            # flattens the disk onto the midplane exactly
            vel[i][1] -= vel[i][1] * (frac * 0.10)
            v_tan = vel[i] - rhat * vel[i].dot(rhat)
            vel[i] -= v_tan * (frac * tang)


@ti.kernel
def k_com_drift(n: ti.i32, n_src: ti.i32):
    """Remove the net momentum of the massive component (keeps scenes centred)."""
    _red[0] = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for j in range(n_src):
        mv = mass[j] * vel[j]
        _red[0] += ti.Vector([mv[0], mv[1], mv[2], mass[j]])
    tm = _red[0][3]
    if tm > 0.0:
        dv = ti.Vector([_red[0][0], _red[0][1], _red[0][2]]) / tm
        for i in range(n):
            vel[i] -= dv


# --- kinematic satellites (moons, rings) ------------------------------------
#
# A single global timestep tuned for year-long planetary orbits is far too
# coarse to N-body-integrate a moon's day-long orbit, or a ring particle's
# even faster one -- both would either fly apart or alias into a strobing
# mess.  So satellites are excluded from gravity entirely (they live past
# scene.n_dynamic, outside the range k_kick/k_drift/k_accel touch) and are
# instead swept along here every frame, parented to their planet's own live
# simulated position.

MAX_SAT = 8000

sat_parent = ti.field(ti.i32, shape=MAX_SAT)
sat_u      = ti.Vector.field(3, ti.f32, shape=MAX_SAT)
sat_v      = ti.Vector.field(3, ti.f32, shape=MAX_SAT)
sat_radius = ti.field(ti.f32, shape=MAX_SAT)
sat_rate   = ti.field(ti.f32, shape=MAX_SAT)
sat_phase0 = ti.field(ti.f32, shape=MAX_SAT)


@ti.kernel
def k_upload_satellites(n_sat: ti.i32,
                         parent: ti.types.ndarray(), u: ti.types.ndarray(),
                         v: ti.types.ndarray(), radius: ti.types.ndarray(),
                         rate: ti.types.ndarray(), phase0: ti.types.ndarray()):
    for i in range(n_sat):
        sat_parent[i] = parent[i]
        sat_u[i] = ti.Vector([u[i, 0], u[i, 1], u[i, 2]])
        sat_v[i] = ti.Vector([v[i, 0], v[i, 1], v[i, 2]])
        sat_radius[i] = radius[i]
        sat_rate[i] = rate[i]
        sat_phase0[i] = phase0[i]


@ti.kernel
def k_update_satellites(n_dyn: ti.i32, n_sat: ti.i32, t: ti.f32):
    for k in range(n_sat):
        i = n_dyn + k
        theta = sat_phase0[k] + sat_rate[k] * t
        off = sat_radius[k] * (ti.cos(theta) * sat_u[k] + ti.sin(theta) * sat_v[k])
        pos[i] = pos[sat_parent[k]] + off


# --- the debris as a gas ----------------------------------------------------
#
# Tidal debris is gas, not gravel.  Drawn as one point sprite per body it never
# stops looking like a swarm of separate objects, however many bodies there are
# and however soft the sprites: the eye picks out individual specks in anything
# but the very densest part, and a star drawn that way is a ball of grit rather
# than something luminous and continuous.
#
# So the bodies are only how the material is *moved*.  What is *drawn* is a
# density field: every frame the particles are splatted into a regular grid
# centred on the hole, smoothed, and handed to the renderer as a 3D texture,
# which the ray marcher then integrates as an emissive, absorbing medium along
# the same bent rays it already uses for everything else.  The star, the
# stream it is drawn out into and the disk it settles into are then literally
# the same substance -- one field, one emission model -- so they blend into
# each other instead of meeting at a seam.
#
# The grid is uniform rather than graded towards the hole.  A graded one buys
# resolution in the inner disk, but it spends it exactly where the ray-marched
# disk already supplies fine structure, and it takes it from the radii where
# the infalling star is -- which is the one place a few cells across is
# obviously not enough.
#
# The extent is a per-scene number rather than a constant, and for the black
# hole it is stretched at spawn time to reach wherever the star is dropped
# from.  A fixed box shows up as a straight edge across the sky as soon as any
# material reaches it, and material past it simply vanishes; the density is
# also faded to nothing over the outermost cells so that the boundary, wherever
# it is, dissolves rather than cuts.
GAS_NX, GAS_NY, GAS_NZ = 160, 56, 160
GAS_RX, GAS_RY = 110.0, 30.0       # default half-extents: wide and flat, which
                                   # is the shape the material usually takes

gas_a = ti.field(ti.f32, shape=(GAS_NZ, GAS_NY, GAS_NX))
gas_b = ti.field(ti.f32, shape=(GAS_NZ, GAS_NY, GAS_NX))
# What actually crosses back to the CPU each frame.  A byte per cell, not a
# float: the field has to be read back and handed to the driver every frame,
# and at this size that copy, not the splat or the smoothing, is the whole
# cost of the technique -- four times the bytes was four times the price for
# precision a glow cannot show.  Square-rooted before quantising, so the
# resolution is spent on the faint material where the eye can see steps.
gas_u8 = ti.field(ti.u8, shape=(GAS_NZ, GAS_NY, GAS_NX))
GAS_DREF = 420.0         # density that saturates the byte: a little above
                         # what the core of an intact star reaches


@ti.kernel
def k_gas_quantise(src: ti.template()):
    for I in ti.grouped(src):
        v = ti.sqrt(ti.min(src[I] * (1.0 / GAS_DREF), 1.0))
        gas_u8[I] = ti.cast(v * 255.0 + 0.5, ti.u8)


@ti.kernel
def k_gas_splat(lo: ti.i32, hi: ti.i32, amp: ti.f32, rx: ti.f32, ry: ti.f32):
    """Trilinear splat of every live particle into the density grid.

    Weighted equally rather than by mass: most of a star is massless tracers
    (see make_star), and they stand for exactly as much material as the few
    that carry the gravity do -- and in the galaxy scene the massive component
    is dark matter, which should not be drawn at all.  Which particles to
    splat is therefore the caller's business, through lo and hi."""
    for I in ti.grouped(gas_a):
        gas_a[I] = 0.0
    for i in range(lo, hi):
        q = pos[i]
        gx = (q[0] / rx * 0.5 + 0.5) * GAS_NX - 0.5
        gy = (q[1] / ry * 0.5 + 0.5) * GAS_NY - 0.5
        gz = (q[2] / rx * 0.5 + 0.5) * GAS_NZ - 0.5
        if (0.0 <= gx) and (gx < GAS_NX - 1) and (0.0 <= gy) and            (gy < GAS_NY - 1) and (0.0 <= gz) and (gz < GAS_NZ - 1):
            ix, iy, iz = int(gx), int(gy), int(gz)
            fx, fy, fz = gx - ix, gy - iy, gz - iz
            for k in ti.static(range(8)):
                dx = ti.static(k & 1)
                dy = ti.static((k >> 1) & 1)
                dz = ti.static((k >> 2) & 1)
                wx = fx if dx == 1 else 1.0 - fx
                wy = fy if dy == 1 else 1.0 - fy
                wz = fz if dz == 1 else 1.0 - fz
                gas_a[iz + dz, iy + dy, ix + dx] += amp * wx * wy * wz


@ti.kernel
def k_gas_blur(src: ti.template(), dst: ti.template(), axis: ti.template()):
    """One separable 1-4-6-4-1 pass.  Three of these turn the splat, which is
    still one particle per cell wherever the material is thin, into something
    with a smoothing length of a couple of cells -- the difference between a
    field that reads as gas and one that reads as a grid of lit voxels."""
    for z, y, x in dst:
        acc = 0.0
        for t in ti.static(range(5)):
            o = t - 2
            sx = x + (o if axis == 0 else 0)
            sy = y + (o if axis == 1 else 0)
            sz = z + (o if axis == 2 else 0)
            w = ti.static([0.0625, 0.25, 0.375, 0.25, 0.0625][t])
            inside = (sx >= 0) and (sx < GAS_NX) and (sy >= 0) and                      (sy < GAS_NY) and (sz >= 0) and (sz < GAS_NZ)
            if inside:
                acc += w * src[sz, sy, sx]
        dst[z, y, x] = acc


def build_gas(scene, amp):
    """Splat, smooth, quantise, and hand back the field for upload.

    The grid is centred on the origin, not on the central body: every scene
    that uses it already pins its barycentre there, and with a heavy star the
    hole itself swings about that point rather than sitting at it."""
    rx, ry = scene.gas_half
    k_gas_splat(scene.gas_lo, scene.n_dynamic, amp, rx, ry)
    k_gas_blur(gas_a, gas_b, 0)
    k_gas_blur(gas_b, gas_a, 1)
    k_gas_blur(gas_a, gas_b, 2)
    k_gas_quantise(gas_b)
    return gas_u8.to_numpy()


# ---------------------------------------------------------------------------
# Scene description
# ---------------------------------------------------------------------------

class Scene:
    """Everything the integrator and the renderer need to know about a setup."""

    def __init__(self, name, pos, vel, mass, soft, col, size, *,
                 n_src=None, gconst=1.0, dt=0.02, substeps=2, pw_rs=0.0,
                 bh=False, disk_in=0.0, disk_out=0.0, recycle=None,
                 cam_dist=100.0, cam_yaw=0.0, cam_pitch=0.35, cam_target=(0, 0, 0),
                 fov=55.0, gain=1.0, world_rs=1.0, kill_drift=True,
                 n_dynamic=None, satellites=None, labels=None, polylines=None,
                 gas="", gas_half=(GAS_RX, GAS_RY), gas_lo=1, extras=()):
        self.name = name
        self.pos = np.ascontiguousarray(pos, dtype=np.float32)
        self.vel = np.ascontiguousarray(vel, dtype=np.float32)
        self.mass = np.ascontiguousarray(mass, dtype=np.float32)
        self.soft = np.ascontiguousarray(soft, dtype=np.float32)
        self.col = np.ascontiguousarray(col, dtype=np.float32)
        self.size = np.ascontiguousarray(size, dtype=np.float32)
        self.n = int(self.pos.shape[0])
        self.n_src = int(n_src if n_src is not None else self.n)
        self.gconst = float(gconst)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.pw_rs = float(pw_rs)
        self.bh = bool(bh)
        self.disk_in = float(disk_in)
        self.disk_out = float(disk_out)
        self.recycle = recycle
        self.cam_dist = float(cam_dist)
        self.cam_yaw = float(cam_yaw)
        self.cam_pitch = float(cam_pitch)
        self.cam_target = np.array(cam_target, dtype=np.float32)
        self.fov = float(fov)
        self.gain = float(gain)
        self.world_rs = float(world_rs)
        self.kill_drift = bool(kill_drift)
        # How this scene can be drawn as gas rather than as separate points.
        #
        #   "volume"  splat the particles into a 3D density grid and ray-march
        #             it as an emitting, absorbing medium.  Needed wherever the
        #             light is bent or the material hides itself, which means
        #             the black hole; it costs a grid readback every frame and
        #             its resolution is whatever the grid is.
        #   "screen"  draw the points, then blur that buffer before the rest of
        #             the pipeline sees it.  A galaxy is optically thin, so
        #             simply adding up its light along the ray is not an
        #             approximation but the right answer -- and doing it in
        #             screen space keeps full pixel resolution and each
        #             particle's own colour, which a grid coarse enough to hold
        #             a whole merger would throw away.
        self.gas = str(gas or "")
        self.gas_half = (float(gas_half[0]), float(gas_half[1]))
        self.gas_lo = int(gas_lo)
        self.attrib = np.empty((self.n, 4), dtype=np.float32)
        self.attrib[:, 0:3] = self.col
        self.attrib[:, 3] = self.size

        # Particles at [n_dynamic, n) are kinematic satellites (moons, ring
        # dust): excluded from gravity, updated by k_update_satellites instead.
        # See the "kinematic satellites" section above for why.
        self.n_dynamic = int(n_dynamic) if n_dynamic is not None else self.n
        if satellites is not None:
            parent, u, v, radius, rate, phase0 = satellites
            self.n_sat = len(parent)
            self.sat_parent = np.ascontiguousarray(parent, dtype=np.int32)
            self.sat_u = np.ascontiguousarray(u, dtype=np.float32)
            self.sat_v = np.ascontiguousarray(v, dtype=np.float32)
            self.sat_radius = np.ascontiguousarray(radius, dtype=np.float32)
            self.sat_rate = np.ascontiguousarray(rate, dtype=np.float32)
            self.sat_phase0 = np.ascontiguousarray(phase0, dtype=np.float32)
        else:
            self.n_sat = 0

        # Particles at [n_dynamic + n_sat, n) are static extras (e.g. the
        # heliosphere shell): uploaded once and never touched by any kernel
        # again, since gravity is bounded to n_dynamic and the satellite
        # kernel to [n_dynamic, n_dynamic + n_sat).  n_base is the vertex
        # count to draw with them hidden -- toggling is then just "how many
        # points to draw" with no extra bookkeeping.
        self.n_base = self.n_dynamic + self.n_sat

        # Named blocks inside the static-extras region, each drawn or not on
        # its own: (name, first index, count).  They live past n_base, so no
        # kernel ever touches them and leaving one out is simply a draw call
        # not made.
        self.extras = [(str(n), int(lo), int(c)) for n, lo, c in extras]

        # (particle_index, display_name) pairs for the optional name-label overlay
        self.labels = list(labels) if labels else []
        # orbit and mission-path polylines: dicts with pos (N,3) f32, color
        # (r,g,b), kind ("orbit" | "mission"), name
        self.polylines = list(polylines) if polylines else []

        # Tidal-disruption state, filled in by the Gargantua preset.  The star
        # pool lives at [1, 1 + star_slots * star_n) but stays outside
        # n_dynamic -- and so outside gravity AND outside the draw call --
        # until a star is actually spawned into it.
        self.star_n = 0          # particles per star
        self.star_src = 0        # of those, how many carry mass
        self.star_slots = 0      # how many stars can exist at once
        self.star_spawned = 0    # how many have been spawned so far
        self.star_cfg = None     # dict of physical parameters for a new star
        self.star_lo = -1        # first index of the most recently dropped star
        self.accreted = 0        # particles the hole has eaten


def _sphere_dirs(n, rng):
    u = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    s = np.sqrt(np.maximum(1.0 - u * u, 0.0))
    return np.stack([s * np.cos(phi), u, s * np.sin(phi)], axis=1)


def plummer_sphere(n, mtot, a, gconst, rng, vscale=1.0, rcut=8.0):
    """Self-consistent Plummer model (Aarseth, Henon and Wielen 1974).

    Sampling the real distribution function matters here: a hand-rolled blob
    would breathe violently for the first few dynamical times and smear out the
    tidal features we are after.
    """
    # Truncated by restricting the mass variable rather than by clipping the
    # radius afterwards.  Clipping does not throw the tail away, it stacks it:
    # every particle beyond the cut lands exactly on it, and for a cut at a few
    # scale radii that is a fifth of the star welded into an infinitely thin
    # shell.  On a star falling towards a hole that shell is the first thing
    # the tide takes, and it leaves as a hollow bubble instead of an envelope.
    xmax = (1.0 + 1.0 / (rcut * rcut)) ** -1.5
    x = rng.uniform(0.0, xmax, n)
    r = a / np.sqrt(np.maximum(x ** (-2.0 / 3.0) - 1.0, 1e-9))
    p = _sphere_dirs(n, rng) * r[:, None]

    q = np.zeros(n)
    todo = np.arange(n)
    while todo.size:
        cand = rng.uniform(0.0, 1.0, todo.size)
        test = rng.uniform(0.0, 0.1, todo.size)
        ok = test < cand * cand * (1.0 - cand * cand) ** 3.5
        q[todo[ok]] = cand[ok]
        todo = todo[~ok]
    vesc = math.sqrt(2.0) * (1.0 + (r / a) ** 2) ** -0.25 * math.sqrt(gconst * mtot / a)
    v = _sphere_dirs(n, rng) * (q * vesc * vscale)[:, None]
    return p, v


def plummer_rho(r, mtot, a):
    return (3.0 * mtot / (4.0 * math.pi * a ** 3)) * (1.0 + (r / a) ** 2) ** -2.5


def total_dphi(r, comps, gconst):
    """dPhi/dr of a superposition of Plummer spheres, comps = [(M, a), ...]."""
    out = np.zeros_like(np.asarray(r, dtype=np.float64))
    for m_i, a_i in comps:
        out += gconst * m_i * r / (r * r + a_i * a_i) ** 1.5
    return out


def jeans_sigma(r, mtot, a, comps, gconst):
    """Isotropic velocity dispersion of a Plummer component embedded in the
    combined potential of `comps`, from the spherical Jeans equation

        rho(r) sigma^2(r) = int_r^inf rho(s) dPhi/ds ds .

    Sampling each component against its own isolated distribution function
    instead -- the obvious shortcut -- leaves the light component far too cold
    inside the heavy one, and the galaxy then violently relaxes and puffs the
    disk up before the encounter even starts.
    """
    grid = np.logspace(math.log10(a * 1e-3), math.log10(a * 300.0), 1024)
    integ = plummer_rho(grid, mtot, a) * total_dphi(grid, comps, gconst)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(grid))])
    tail = cum[-1] - cum
    s2 = np.interp(r, grid, tail) / np.maximum(plummer_rho(r, mtot, a), 1e-300)
    return np.sqrt(np.maximum(s2, 0.0))


def plummer_component(n, mtot, a, comps, gconst, rng, rcut=8.0):
    """Plummer-profile positions with Jeans-consistent isotropic velocities."""
    r = np.empty(n)
    todo = np.arange(n)
    while todo.size:                       # resample rather than clip: clipping
        x = rng.uniform(0.0, 1.0, todo.size)   # piles a shell up at exactly rcut
        cand = a / np.sqrt(np.maximum(x ** (-2.0 / 3.0) - 1.0, 1e-12))
        ok = cand <= rcut * a
        r[todo[ok]] = cand[ok]
        todo = todo[~ok]
    p = _sphere_dirs(n, rng) * r[:, None]
    sig = jeans_sigma(r, mtot, a, comps, gconst)
    v = rng.normal(0.0, 1.0, (n, 3)) * sig[:, None]
    return p, v


def _rot_matrix(incl, node):
    """Tilt a disk that starts in the x-z plane: incline about x, then swing the
    line of nodes about y."""
    ci, si = math.cos(incl), math.sin(incl)
    cn, sn = math.cos(node), math.sin(node)
    rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]], dtype=np.float64)
    ry = np.array([[cn, 0, sn], [0, 1, 0], [-sn, 0, cn]], dtype=np.float64)
    return ry @ rx


def two_body_orbit(gconst, mtot, sep, a_orb, r_peri):
    """Radial and tangential relative speed for a bound orbit of semi-major axis
    a_orb and pericentre r_peri, evaluated at separation sep."""
    ecc = max(0.0, 1.0 - r_peri / a_orb)
    ell = math.sqrt(max(gconst * mtot * a_orb * (1.0 - ecc * ecc), 0.0))
    v2 = max(gconst * mtot * (2.0 / sep - 1.0 / a_orb), 0.0)
    v_t = ell / sep
    v_r = -math.sqrt(max(v2 - v_t * v_t, 0.0))
    return v_r, v_t


def _kepler_rot_coeffs(i_deg, node_deg, argp_deg):
    """The perifocal -> reference-plane rotation matrix (Vallado, 'Fundamentals
    of Astrodynamics and Applications'), reduced to the 6 coefficients that
    turn a 2D point in the orbital plane into 3D -- the third row is dropped
    below because it is only needed for out-of-plane (z) motion here."""
    i, node, argp = math.radians(i_deg), math.radians(node_deg), math.radians(argp_deg)
    ci, si = math.cos(i), math.sin(i)
    cn, sn = math.cos(node), math.sin(node)
    cw, sw = math.cos(argp), math.sin(argp)
    r11, r12 = cn * cw - sn * sw * ci, -cn * sw - sn * cw * ci
    r21, r22 = sn * cw + cn * sw * ci, -sn * sw + cn * cw * ci
    r31, r32 = sw * si, cw * si
    return r11, r12, r21, r22, r31, r32


def kepler_state(a, e, i_deg, node_deg, argp_deg, nu, gm=1.0):
    """Classical orbital elements plus a true anomaly -> Cartesian (pos, vel),
    via the standard perifocal-to-reference-frame rotation.  a is in sim
    length units, angles in degrees, nu (true anomaly) in radians, gm = G*M of
    the body being orbited.  Returns (pos, vel) in the sim's (x, y-up, z)
    convention: sim x/z is the reference (ecliptic) plane, sim y is the pole
    -- matching every circular orbit already built by hand elsewhere in this
    file, so a body placed this way drops into the same picture seamlessly.

    This -- real eccentricity, inclination, and orientation per body, vis-viva
    speed instead of a flat circular one -- is what "very accurate physics"
    means for a solar-system sim: planets now visibly speed up at perihelion
    and slow down at aphelion, and Pluto's tilted, eccentric orbit genuinely
    dips inside Neptune's near perihelion, the way the real one does.
    """
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    x_pf, y_pf = r * math.cos(nu), r * math.sin(nu)
    h = math.sqrt(max(gm * p, 1e-30))
    vx_pf, vy_pf = -gm / h * math.sin(nu), gm / h * (e + math.cos(nu))

    r11, r12, r21, r22, r31, r32 = _kepler_rot_coeffs(i_deg, node_deg, argp_deg)
    Xa, Ya = r11 * x_pf + r12 * y_pf, r21 * x_pf + r22 * y_pf
    Za = r31 * x_pf + r32 * y_pf
    Vxa, Vya = r11 * vx_pf + r12 * vy_pf, r21 * vx_pf + r22 * vy_pf
    Vza = r31 * vx_pf + r32 * vy_pf
    return np.array([Xa, Za, Ya]), np.array([Vxa, Vza, Vya])


def kepler_orbit_outline(a, e, i_deg, node_deg, argp_deg, n=192):
    """Sample n points around the full ellipse (position only) for drawing an
    orbit-path line -- vectorised over true anomaly for a single fast call."""
    nu = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    x_pf, y_pf = r * np.cos(nu), r * np.sin(nu)
    r11, r12, r21, r22, r31, r32 = _kepler_rot_coeffs(i_deg, node_deg, argp_deg)
    Xa, Ya = r11 * x_pf + r12 * y_pf, r21 * x_pf + r22 * y_pf
    Za = r31 * x_pf + r32 * y_pf
    return np.stack([Xa, Za, Ya], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 1) Solar System disk
# ---------------------------------------------------------------------------

def catmull_rom(points, samples_per_seg=24):
    """Smooth C1 path through 3D control points (Nx3) -- duplicates the end
    points so the curve starts and ends exactly on the first/last waypoint.
    Used for the illustrative spacecraft trajectories below: real mission
    paths are patched-conic, gravity-assisted trajectories that would need a
    full ephemeris and launch-date epoch to reproduce exactly, which this
    sim's timeless N-body model has no notion of.  What IS kept faithful is
    the real sequence and heliocentric distance of every flyby and encounter;
    only the smooth curve connecting them is stylised."""
    pts = np.asarray(points, dtype=np.float64)
    pts = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-2])
    return np.array(out, dtype=np.float32)


def _helio_wp(r_au, theta_deg, incl_deg=0.0):
    """A mission waypoint: r_au is the TRUE heliocentric distance in AU (so
    real published mission facts, e.g. "Voyager 1 crossed the heliopause at
    121.6 AU", plug straight in), theta_deg the azimuth, incl_deg the angle
    above (+) or below (-) the ecliptic."""
    r = r_au * 10.0
    th, inc = math.radians(theta_deg), math.radians(incl_deg)
    xz = r * math.cos(inc)
    return [xz * math.cos(th), r * math.sin(inc), xz * math.sin(th)]


def _mission_paths():
    """Illustrative real mission trajectories: correct sequence and distance
    of every flyby, smoothly interpolated between them (see catmull_rom)."""
    w = _helio_wp
    missions = [
        ("Voyager 1", (0.16, 0.46, 0.52), [
            w(1.0, 0, 0), w(5.2, 35, 3), w(9.5, 65, 8), w(25, 85, 20),
            w(60, 100, 28), w(100, 112, 32), w(121.6, 118, 34), w(165, 122, 35),
        ]),
        ("Voyager 2", (0.48, 0.20, 0.48), [
            w(1.0, -10, 0), w(5.2, -35, -2), w(9.5, -60, -10), w(19.2, -85, -25),
            w(30.1, -105, -38), w(60, -118, -44), w(100, -126, -47),
            w(119.0, -130, -48), w(140, -133, -48),
        ]),
        ("New Horizons", (0.52, 0.30, 0.10), [
            w(1.0, 15, 0), w(5.2, 25, 0.5), w(15, 40, 1), w(25, 52, 1.5),
            w(33.0, 60, 2), w(43.4, 64, 2.3), w(58, 67, 2.6),
        ]),
        ("Pioneer 10", (0.24, 0.48, 0.24), [
            w(1.0, -150, 0), w(2.8, -145, 0.5), w(5.2, -138, 1),
            w(20, -128, 2), w(50, -120, 3), w(80, -115, 3.5),
        ]),
        ("Pioneer 11", (0.40, 0.46, 0.17), [
            w(1.0, -160, 0), w(5.2, -150, 1), w(9.5, -135, 6),
            w(30, -120, 14), w(44, -114, 16),
        ]),
        ("Cassini", (0.50, 0.42, 0.14), [
            w(0.95, 170, 0), w(0.72, 160, 2), w(0.72, 178, -2), w(1.0, -170, 0),
            w(2.5, -155, 0.5), w(5.2, -140, 1), w(9.0, -122, 1.2),
            w(9.5, -118, 1.3), w(9.6, -113, 2.0), w(9.3, -121, 0.5), w(9.5, -117, 1.0),
        ]),
        ("Juno", (0.24, 0.34, 0.48), [
            w(1.0, -60, 0), w(1.0, -50, 1), w(3.0, -40, 2), w(5.2, -25, 3),
            w(5.35, -20, 8), w(5.1, -30, -6), w(5.2, -25, 3),
        ]),
        ("Parker Solar Probe", (0.55, 0.17, 0.09), [
            w(1.0, 200, 0), w(0.72, 210, 1), w(0.166, 220, 0), w(0.72, 232, -1),
            w(0.095, 245, 0), w(0.72, 258, 1), w(0.062, 270, 0),
            w(0.72, 282, -1), w(0.045, 295, 0), w(0.3, 305, 0),
        ]),
    ]
    return [{"name": name, "kind": "mission", "color": color,
            "pos": catmull_rom(wps, samples_per_seg=20)}
            for name, color, wps in missions]


def preset_solar_system(rng):
    """The Sun, the eight planets, the five IAU dwarf planets, their major
    moons, Saturn's rings, the asteroid and Kuiper belts, a thick Oort cloud
    shell, the heliosphere, and a handful of real space-mission trajectories.

    Every named body (planet or dwarf planet) is placed on its REAL Keplerian
    orbit -- true eccentricity, inclination, and orientation, not a flattened
    circle -- via kepler_state(), which also gives it the correct vis-viva
    speed (faster at perihelion, slower at aphelion).  This is what makes
    Pluto's tilted, eccentric orbit actually dip inside Neptune's near
    perihelion, the way the real one does (see kepler_state's docstring for
    the numbers).  The asteroid and Kuiper belts stay statistical (randomly
    scattered populations, not individually tracked bodies), which is the
    standard, defensible approximation for a debris population.

    Moons and ring dust are NOT part of the N-body gravity pass -- a global
    timestep tuned for year-long planetary orbits is far too coarse to
    integrate a day-long moon orbit (let alone a ring particle's faster one)
    without it flying apart or aliasing into a strobing mess.  Instead they
    are kinematic satellites (see the "kinematic satellites" Taichi section):
    parented to their planet's live simulated position every frame, with an
    angular rate set as a multiple of that planet's own orbital rate so
    closer moons visibly move faster than farther ones, the way real moons do.

    Phobos and Deimos are both given PROGRADE orbits deliberately: neither
    real moon is retrograde (both orbit in the same sense as Mars's own
    rotation, in its equatorial plane).  Phobos's real oddity is that it
    orbits faster than Mars rotates, so it rises in the west and sets in the
    east -- captured here as an unusually fast rate, not a reversed one,
    since flipping it would trade one accurate detail for a wrong one.
    """
    G = 1.0
    m_sun = 1.0
    # name, semi-major axis (AU), eccentricity, inclination/node/argument of
    # perihelion (degrees, real orbital elements), mass (solar masses), draw
    # radius, colour, moons as (orbit radius in multiples of the planet's own
    # drawn radius, angular rate in multiples of the planet's own orbital
    # rate, drawn size, colour, orbital inclination in degrees, retrograde)
    bodies = [
        ("Mercury", 0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645,
         1.66e-7, 0.45, (1.00, 0.86, 0.66), []),
        ("Venus", 0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298,
         2.45e-6, 0.70, (1.00, 0.82, 0.50), []),
        ("Earth", 1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719,
         3.00e-6, 0.72, (0.45, 0.68, 1.00), [
            (1.9, 9.0, 0.11, (0.80, 0.80, 0.78), 8.0, False),                 # Moon
        ]),
        ("Mars", 1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084,
         3.23e-7, 0.58, (1.00, 0.48, 0.32), [
            (1.5, 22.0, 0.045, (0.65, 0.55, 0.48), 4.0, False),               # Phobos (prograde, real)
            (2.1, 12.0, 0.040, (0.60, 0.58, 0.55), 26.0, False),              # Deimos (prograde, real)
        ]),
        ("Ceres", 2.7691651, 0.0760090, 10.59406, 80.30553, 73.59764,
         4.72e-10, 0.14, (0.62, 0.60, 0.58), []),
        ("Jupiter", 5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385,
         9.55e-4, 1.70, (0.95, 0.80, 0.62), [
            (2.3, 12.0, 0.15, (0.90, 0.78, 0.45), 3.0, False),                # Io
            (3.0, 9.0, 0.13, (0.85, 0.80, 0.72), 5.0, False),                 # Europa
            (3.9, 6.5, 0.17, (0.72, 0.66, 0.58), 7.0, False),                 # Ganymede
            (5.0, 4.5, 0.16, (0.48, 0.44, 0.42), 9.0, False),                 # Callisto
        ]),
        ("Saturn", 9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194,
         2.86e-4, 1.50, (0.98, 0.88, 0.66), [
            (2.6, 14.0, 0.075, (0.92, 0.94, 0.96), 15.0, False),              # Enceladus
            (3.4, 10.0, 0.095, (0.80, 0.78, 0.72), 18.0, False),              # Rhea
            (4.6, 7.0, 0.19, (0.90, 0.72, 0.42), 22.0, False),                # Titan
        ]),
        ("Uranus", 19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424,
         4.37e-5, 1.05, (0.62, 0.92, 0.96), [
            (2.6, 8.0, 0.09, (0.72, 0.78, 0.85), 12.0, False),                # Titania
        ]),
        ("Neptune", 30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135,
         5.15e-5, 1.02, (0.42, 0.60, 1.00), [
            (2.8, 9.0, 0.10, (0.62, 0.72, 0.95), 20.0, True),                 # Triton (retrograde, real)
        ]),
        ("Pluto", 39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676,
         6.55e-9, 0.26, (0.80, 0.68, 0.58), [
            (2.2, 7.0, 0.13, (0.72, 0.70, 0.68), 6.0, False),                 # Charon
        ]),
        ("Haumea", 43.13, 0.19126, 28.2137, 121.900, 239.041,
         2.02e-9, 0.19, (0.88, 0.90, 0.92), []),
        ("Makemake", 45.79, 0.16126, 28.9835, 79.620, 294.834,
         1.56e-9, 0.18, (0.72, 0.48, 0.38), []),
        ("Eris", 67.78, 0.43607, 44.0445, 35.9531, 151.639,
         8.35e-9, 0.25, (0.85, 0.85, 0.88), []),
    ]
    P, V, M, S, C, R = [], [], [], [], [], []
    labels = [(0, "Sun")]

    P.append([0.0, 0.0, 0.0]); V.append([0.0, 0.0, 0.0])
    M.append(m_sun); S.append(0.6); C.append([4.0, 3.1, 1.8]); R.append(2.6)

    moon_specs = []   # (parent_idx, planet_a, planet_rad, radius_mult, rate_mult, size, colour, incl_deg, retro)
    saturn_idx = saturn_a = saturn_rad = None
    orbit_lines = []
    for name, a_au, e, i_deg, node_deg, argp_deg, m, rad, c, moons in bodies:
        a = a_au * 10.0
        nu0 = rng.uniform(0.0, 2.0 * math.pi)
        p_vec, v_vec = kepler_state(a, e, i_deg, node_deg, argp_deg, nu0, gm=G * m_sun)
        P.append(p_vec.tolist()); V.append(v_vec.tolist())
        M.append(m); S.append(0.25)
        C.append([c[0] * 1.6, c[1] * 1.6, c[2] * 1.6]); R.append(rad)
        parent_idx = len(P) - 1
        labels.append((parent_idx, name))
        orbit_lines.append({"name": name, "kind": "orbit",
                            "color": (c[0] * 0.55, c[1] * 0.55, c[2] * 0.55),
                            "pos": kepler_orbit_outline(a, e, i_deg, node_deg, argp_deg)})
        for radius_mult, rate_mult, size, mc, incl_deg, retro in moons:
            moon_specs.append((parent_idx, a, rad, radius_mult, rate_mult, size, mc, incl_deg, retro))
        if name == "Saturn":
            saturn_idx, saturn_a, saturn_rad = parent_idx, a, rad

    n_src = len(P)

    def belt(count, r0, r1, thick, tint, rad, ecc=0.02):
        u = rng.uniform(0.0, 1.0, count)
        r = np.sqrt(r0 * r0 + u * (r1 * r1 - r0 * r0))
        ph = rng.uniform(0, 2 * math.pi, count)
        vc = np.sqrt(G * m_sun / r) * (1.0 + rng.normal(0, ecc, count))
        P.extend(np.stack([r * np.cos(ph), rng.normal(0, thick, count),
                           r * np.sin(ph)], axis=1).tolist())
        V.extend(np.stack([-vc * np.sin(ph),
                           rng.normal(0, 0.004, count) * vc,
                           vc * np.cos(ph)], axis=1).tolist())
        M.extend([0.0] * count); S.extend([0.05] * count); R.extend([rad] * count)
        shade = rng.uniform(0.55, 1.35, count)[:, None]
        C.extend((np.array(tint)[None, :] * shade).tolist())

    belt(2200, 21.0, 33.0, 0.75, (0.85, 0.80, 0.70), 0.26)    # asteroid belt
    belt(3000, 320.0, 460.0, 6.0, (0.60, 0.78, 1.00), 0.85)   # Kuiper belt

    # Oort cloud: a vast, near-isotropic spherical shell of icy debris well
    # beyond Kuiper.  Unlike every belt above it is NOT confined to the
    # ecliptic plane -- that is exactly what makes it read as the edge of the
    # system rather than just another ring.  Pushed out past every mission
    # path and the heliopause (below) so the ordering stays true to reality --
    # nothing humanity has launched has come remotely close to it -- and
    # scaled in for visual reach (the real Oort cloud starts around 2000 AU)
    # rather than kept to scale.
    def oort_cloud(count, r0, r1, tint, rad):
        u = rng.uniform(0.0, 1.0, count)
        r = np.cbrt(r0 ** 3 + u * (r1 ** 3 - r0 ** 3))     # uniform in volume
        dirs = _sphere_dirs(count, rng)
        pos_ = dirs * r[:, None]
        # Held still rather than orbited.  A body four thousand units out has
        # an orbital period some thousands of times longer than the run, so
        # integrating it buys a motion of well under a pixel while costing a
        # tenth of the scene's particle budget every step.
        P.extend(pos_.tolist())
        V.extend([[0.0, 0.0, 0.0]] * count)
        M.extend([0.0] * count); S.extend([0.05] * count); R.extend([rad] * count)
        shade = rng.uniform(0.5, 1.3, count)[:, None]
        C.extend((np.array(tint)[None, :] * shade).tolist())

    n_dynamic = len(P)   # [n_dynamic, n) below are kinematic satellites, not gravitating

    sat_parent, sat_u, sat_v, sat_radius, sat_rate, sat_phase0 = [], [], [], [], [], []

    def add_moon(parent_idx, planet_a, planet_rad, radius_mult, rate_mult, size, color, incl_deg, retro):
        omega_planet = math.sqrt(G * m_sun / planet_a) / planet_a
        rot = _rot_matrix(math.radians(incl_deg), rng.uniform(0, 2 * math.pi))
        P.append([0.0, 0.0, 0.0]); V.append([0.0, 0.0, 0.0])
        M.append(0.0); S.append(0.03)
        C.append([color[0] * 1.9, color[1] * 1.9, color[2] * 1.9]); R.append(size)
        sat_parent.append(parent_idx)
        sat_u.append(rot @ np.array([1.0, 0.0, 0.0]))
        sat_v.append(rot @ np.array([0.0, 0.0, 1.0]))
        sat_radius.append(radius_mult * planet_rad)
        sat_rate.append((-1.0 if retro else 1.0) * omega_planet * rate_mult)
        sat_phase0.append(rng.uniform(0, 2 * math.pi))

    for spec in moon_specs:
        add_moon(*spec)

    def saturn_ring(parent_idx, planet_a, planet_rad, count):
        """Many independently-phased kinematic tracers rather than a static
        disk, so the ring visibly differentially-rotates the way a real one
        does -- inner material laps the outer edge many times over."""
        r0, r1 = 1.55 * planet_rad, 2.30 * planet_rad
        gap_lo, gap_hi = 1.86 * planet_rad, 1.95 * planet_rad   # Cassini-division-style gap
        r = r0 + rng.uniform(0.0, 1.0, count) * (r1 - r0)
        r = r[~((r > gap_lo) & (r < gap_hi))]
        n = r.shape[0]
        omega_planet = math.sqrt(G * m_sun / planet_a) / planet_a
        rate = omega_planet * 10.0 * (r0 / r) ** 1.5
        rot = _rot_matrix(math.radians(26.7), math.radians(35.0))   # Saturn's real obliquity
        u_ax, v_ax = rot @ np.array([1.0, 0.0, 0.0]), rot @ np.array([0.0, 0.0, 1.0])
        band = np.clip((r - r0) / (r1 - r0), 0.0, 1.0)
        shade = (0.55 + 0.55 * rng.uniform(0.0, 1.0, n)) * (1.0 - 0.35 * np.sin(band * 9.0) ** 2)
        tint = np.array([0.86, 0.80, 0.62])
        P.extend([[0.0, 0.0, 0.0]] * n); V.extend([[0.0, 0.0, 0.0]] * n)
        M.extend([0.0] * n); S.extend([0.02] * n)
        C.extend((tint[None, :] * shade[:, None] * 0.55).tolist())
        R.extend((0.032 + 0.020 * rng.uniform(0.0, 1.0, n)).tolist())
        sat_parent.extend([parent_idx] * n)
        sat_u.extend([u_ax] * n); sat_v.extend([v_ax] * n)
        sat_radius.extend(r.tolist()); sat_rate.extend(rate.tolist())
        sat_phase0.extend(rng.uniform(0, 2 * math.pi, n).tolist())

    saturn_ring(saturn_idx, saturn_a, saturn_rad, 4000)

    satellites = (sat_parent, np.array(sat_u), np.array(sat_v), sat_radius, sat_rate, sat_phase0)

    # Heliosphere / heliopause: a static (non-orbiting -- it is a plasma
    # standoff boundary shaped by the solar wind, not gravitating debris)
    # shell, compressed toward the nose (the direction the Sun is currently
    # moving through the local interstellar medium) and drawn into a long
    # tail on the far side, matching the classic comet-shaped picture.  Real
    # distances: nose/heliopause ~120 AU (both Voyagers crossed within a few
    # AU of that), flanks ~100 AU, tail poorly constrained but modelled here
    # as ~180 AU -- comfortably inside the Oort cloud shell above, as it
    # should be.
    def heliosphere_shell(count, nose_au, side_au, tail_au, tint, rad):
        dirs = _sphere_dirs(count, rng)
        fx = dirs[:, 0]
        r_au = np.where(fx >= 0.0,
                        side_au + (nose_au - side_au) * fx,
                        side_au + (tail_au - side_au) * (-fx))
        pos_ = dirs * (r_au * 10.0)[:, None]
        M.extend([0.0] * count); S.extend([0.05] * count); R.extend([rad] * count)
        shade = rng.uniform(0.45, 1.0, count)[:, None]
        C.extend((np.array(tint)[None, :] * shade).tolist())
        return pos_

    # Both shells are switchable extras, so they sit past everything the
    # kernels touch.  Draw radius is large (not to scale with anything else)
    # on purpose: at the camera distances where they are actually in frame, a
    # physically-scaled point is sub-pixel and the sprite shader's distance
    # falloff fades it to nothing well before that, so these are sized to read
    # as a haze from out there.
    extras = []
    oort_lo = len(P)
    oort_cloud(9000, 2200.0, 4600.0, (0.74, 0.84, 1.00), 16.0)
    extras.append(("Oort cloud", oort_lo, len(P) - oort_lo))

    helio_lo = len(P)
    helio_pos = heliosphere_shell(3200, 120.0, 100.0, 180.0, (0.45, 0.65, 1.00), 11.0)
    P.extend(helio_pos.tolist())
    V.extend([[0.0, 0.0, 0.0]] * int(helio_pos.shape[0]))
    extras.append(("heliosphere / heliopause", helio_lo, len(P) - helio_lo))

    polylines = orbit_lines + _mission_paths()

    return Scene("Solar System", P, V, M, S, C, R,
                 n_src=n_src, n_dynamic=n_dynamic, satellites=satellites,
                 labels=labels, polylines=polylines, extras=extras,
                 gconst=G, dt=0.06, substeps=3,
                 cam_dist=150.0, cam_pitch=0.42, gain=1.6, kill_drift=False)


# ---------------------------------------------------------------------------
# 2) Galaxy merger -- live halos, tracer disks, Toomre-style plunging orbit
# ---------------------------------------------------------------------------

def preset_galaxy_merger(rng):
    """Two disk galaxies on a grazing, bound encounter.

    Each galaxy is a live two-component N-body system -- a bulge holding most
    of the mass plus a lighter extended halo -- carrying a disk of massless
    tracers.  Three details decide whether this looks like a merger or like two
    smudges colliding:

    * The disks must be PROGRADE (disk spin aligned with the orbital angular
      momentum).  Retrograde disks are merely heated and throw no tails at all.
    * The bulge has to be extended enough (a = 7 kpc) that its own two-body
      relaxation time stays far longer than the run.  A compact bulge made of a
      few hundred particles dissolves in ~25 time units, draining the central
      potential and puffing the disk into a fuzzball before pericentre.
    * The halo has to stay light.  A massive, extended halo binds the stripped
      material, so the tails never get away, and it sinks the orbit so fast
      that the pair coalesces on the first passage.

    With those set, pericentre at ~2x the disk radius gives the classic
    sequence: bridge, two long tidal tails, a second passage, then coalescence
    into a spheroidal remnant.
    """
    G = 1.0
    m_bulge, a_bulge, soft_bulge = 1100.0, 7.0, 2.0
    m_halo, a_halo, soft_halo = 300.0, 25.0, 3.0
    m_gal = m_bulge + m_halo

    n_heavy = max(300, ARGS.halo)
    n_bulge = max(80, int(n_heavy * 0.7))
    n_halo = n_heavy - n_bulge
    n_disk = max(500, min(ARGS.disk, (MAX_N - 2 * n_heavy) // 2))

    r_d, r_min, r_max, h_z = 6.0, 2.0, 24.0, 0.45
    n_arms, arm_strength = 2, 0.62

    # Pericentre at ~2x the disk radius: deep enough that the tidal radius cuts
    # into the outer disk, shallow enough that the material leaves as coherent
    # streams rather than an expanding shell.
    sep, a_orb, r_peri = 150.0, 140.0, 45.0
    v_r, v_t = two_body_orbit(G, 2.0 * m_gal, sep, a_orb, r_peri)
    centres = [np.array([-0.5 * sep, 0.0, 0.0]), np.array([0.5 * sep, 0.0, 0.0])]
    bulk = [np.array([-0.5 * v_r, 0.0, 0.5 * v_t]),
            np.array([0.5 * v_r, 0.0, -0.5 * v_t])]

    # (inclination, line of nodes, spin sense, tint).  The orbit lies in the x-z
    # plane with its angular momentum along +y, and spin = -1 puts the disk
    # angular momentum along +y too, i.e. prograde.
    geom = [(math.radians(10.0), math.radians(0.0),  -1.0, (0.32, 0.55, 1.00)),
            (math.radians(30.0), math.radians(70.0), -1.0, (1.00, 0.38, 0.22))]

    hv_p, hv_v, hv_m, hv_s, disk_p, disk_v, disk_c = [], [], [], [], [], [], []

    def v_circ(r):
        # v_c^2 = r dPhi/dr for a sum of Plummer spheres
        return np.sqrt(G * (m_bulge * r * r / (r * r + a_bulge ** 2) ** 1.5
                            + m_halo * r * r / (r * r + a_halo ** 2) ** 1.5))

    for gi in range(2):
        incl, node, spin, tint = geom[gi]
        rot = _rot_matrix(incl, node)

        comps = [(m_bulge, a_bulge), (m_halo, a_halo)]
        bp, bv = plummer_component(n_bulge, m_bulge, a_bulge, comps, G, rng, rcut=6.0)
        hp, hp_v = plummer_component(n_halo, m_halo, a_halo, comps, G, rng, rcut=4.0)
        hv_p += [bp + centres[gi], hp + centres[gi]]
        hv_v += [bv + bulk[gi], hp_v + bulk[gi]]
        hv_m += [np.full(n_bulge, m_bulge / n_bulge),
                 np.full(n_halo, m_halo / n_halo)]
        hv_s += [np.full(n_bulge, soft_bulge), np.full(n_halo, soft_halo)]

        # exponential surface density: Sigma ~ exp(-r/r_d) => p(r) ~ r exp(-r/r_d)
        r = rng.gamma(2.0, r_d, n_disk)
        bad = (r < r_min) | (r > r_max)
        while bad.any():
            r[bad] = rng.gamma(2.0, r_d, int(bad.sum()))
            bad = (r < r_min) | (r > r_max)
        ph = rng.uniform(0, 2 * math.pi, n_disk)
        z = rng.laplace(0.0, h_z, n_disk)

        # Two-armed logarithmic spiral, put in by hand.  A cold exponential
        # disk of test particles will not grow arms on its own here: the
        # tracers are massless, so there is no self-gravity in the disk for a
        # density wave to ride on, and even with one the pattern would take
        # longer to appear than the encounter takes to happen.  So the arms are
        # part of the initial condition -- each particle is pulled part of the
        # way from where it was towards the nearest arm, at fixed radius, which
        # leaves the orbits untouched and only moves material around in
        # azimuth.  Differential rotation then winds them up, and the encounter
        # tears them apart, both of which are real.
        pitch = math.radians(20.0)
        phi_arm = np.log(r / r_min) / math.tan(pitch)
        off = np.angle(np.exp(1j * n_arms * (ph - phi_arm))) / n_arms
        ph = ph - arm_strength * off
        # arms are where the star formation is, so make them brighter and bluer
        arm = np.exp(-((off * n_arms) ** 2) * 1.6)

        v_c = v_circ(r)
        sig = 0.09 * v_c                      # cold-ish disk, Toomre Q ~ 1.5

        px = np.stack([r * np.cos(ph), z, r * np.sin(ph)], axis=1)
        vx = np.stack([-spin * v_c * np.sin(ph),
                       np.zeros(n_disk),
                       spin * v_c * np.cos(ph)], axis=1)
        vx += rng.normal(0.0, 1.0, (n_disk, 3)) * sig[:, None] * np.array([1.0, 0.6, 1.0])

        disk_p.append(px @ rot.T + centres[gi])
        disk_v.append(vx @ rot.T + bulk[gi])

        t = np.clip(r / r_max, 0.0, 1.0)[:, None]
        base = np.array(tint)[None, :]
        core = np.array([1.0, 0.95, 0.90])[None, :]
        shade = rng.uniform(0.65, 1.45, n_disk)[:, None] * (1.0 + 0.9 * arm)[:, None]
        disk_c.append((core * (1.0 - t) ** 9 * 0.9 + base * (0.85 + 0.45 * t))
                      * shade * 0.85)

    P = np.concatenate(hv_p + disk_p)
    V = np.concatenate(hv_v + disk_v)
    n_src = 2 * n_heavy
    n_tot = P.shape[0]
    M = np.zeros(n_tot); M[:n_src] = np.concatenate(hv_m)
    S = np.zeros(n_tot); S[:n_src] = np.concatenate(hv_s); S[n_src:] = 0.4
    C = np.zeros((n_tot, 3)); C[n_src:] = np.concatenate(disk_c)
    R = np.zeros(n_tot); R[n_src:] = 0.40      # dark matter particles are invisible

    return Scene("Galaxy Merger", P, V, M, S, C, R,
                 n_src=n_src, gconst=G, dt=0.04, substeps=2,
                 cam_dist=285.0, cam_pitch=1.30, cam_yaw=0.0, gain=1.0,
                 # only the disks become gas; the massive component here is
                 # dark matter and a halo drawn as a glowing cloud would be
                 # exactly the wrong picture of it
                 gas="screen", gas_lo=n_src)


# ---------------------------------------------------------------------------
# 3) Stellar-mass black hole -- geometric units, 1 world unit == 1 r_s
# ---------------------------------------------------------------------------

def preset_black_hole(rng):
    """A bare 10-solar-mass Schwarzschild hole -- no disk until you make one.

    Press X (or use the Tidal Disruption panel) to drop a star onto it.  The
    star is a live, self-gravitating Plummer ball on a bound, eccentric orbit
    whose pericentre sits inside its own tidal radius, so it survives the fall
    in, gets stretched into a stream at pericentre, and that stream is what
    becomes the disk.

    One honest caveat about the star.  It is drawn several times the size of
    the hole, which is the right way round -- the hole is 30 km across and any
    real star dwarfs it -- but nowhere near the true ratio: a sun-like star
    next to a 10-solar-mass hole is some twenty thousand r_s across, so at this
    zoom it would fill the sky and the photon ring would be a speck.  The same
    scale problem is why a real star that size is torn apart tens of thousands
    of r_s out, far outside anywhere the lensing is visible, which in turn is
    why observed tidal disruptions are events around supermassive holes, and
    why real 10-solar-mass holes -- Cygnus X-1 and the rest of the X-ray
    binaries -- get their disks from a companion feeding them rather than from
    one swallowed star.  So the star here is deliberately compact for its mass:
    big enough on screen to read as the larger body, small enough that the
    disruption happens where you can watch it against the ring.  "Sustain
    disk" models the companion-fed case, and stops the disk emptying out.
    """
    G = 1.0
    rs = 1.0
    gm = 0.5 * rs            # c = 1 and r_s = 2GM, so GM = r_s / 2

    # A stretched stream is a one-dimensional object drawn out over a hundred
    # r_s, so it needs far more particles than a ball does to stay continuous
    # rather than reading as a scatter of specks.  Only a coarse subset of them
    # carries the star's mass, though: self-gravity is the one O(N^2) term in
    # the whole simulation, and resolving the star's potential needs a few
    # thousand bodies where drawing its stream needs tens of thousands.  The
    # rest follow the same distribution function as massless tracers, so they
    # trace exactly the orbits the massive ones do, at O(N) instead.
    star_n = 36000           # particles drawn per star
    star_src = 900           # of those, the ones that carry its mass
    star_slots = 2           # stars that can be on the board at once

    P = [[0.0, 0.0, 0.0]]
    V = [[0.0, 0.0, 0.0]]
    M = [gm / G]
    S = [1e-3]
    C = [[0.0, 0.0, 0.0]]
    R = [0.0]                # the hole itself is drawn by the ray marcher

    # The dormant star pool.  Parked out at the graveyard radius so that even
    # if something draws them they are far past the point where a point sprite
    # fades to nothing; they carry no mass and sit outside n_dynamic, so they
    # are inert until spawned.
    pool = star_n * star_slots
    P.extend([[GRAVEYARD, GRAVEYARD, GRAVEYARD]] * pool)
    V.extend([[0.0, 0.0, 0.0]] * pool)
    M.extend([0.0] * pool)
    S.extend([0.05] * pool)
    C.extend([[1.0, 0.95, 0.85]] * pool)
    R.extend([0.030] * pool)

    sc = Scene("Black Hole", P, V, M, S, C, R,
               n_src=1, n_dynamic=1, gconst=G, dt=0.16, substeps=4, pw_rs=rs,
               bh=True, disk_in=3.0, disk_out=26.0, recycle=None,
               cam_dist=52.0, cam_pitch=0.30, cam_yaw=0.6,
               fov=42.0, gain=0.075, world_rs=rs, kill_drift=False,
               gas="volume", gas_half=(GAS_RX, GAS_RY), gas_lo=1)
    sc.star_n = star_n
    sc.star_src = star_src
    sc.star_slots = star_slots
    sc.star_cfg = {
        "m_star": 1.0e-3,
        "r_star": 7.0,       # outer radius in r_s -- see the docstring.  Well
                             # clear of the 2.6 r_s shadow, so the star reads
                             # as the larger of the two bodies on screen, which
                             # is what it is: the hole here is 30 km across.
        "r_apo": 150.0,      # apocentre of the infall orbit
        "r_start": 96.0,     # where it is actually dropped -- see make_star
        "r_peri": 17.0,      # pericentre, deep inside the tidal radius
        "soft": 0.05,
        # softening for the massive subset: about the spacing between them in
        # the star's core, so they model a smooth potential rather than
        # scattering off one another
        "soft_src": 0.30,
        "gm": gm,
        "rs": rs,
    }
    return sc


def make_star(cfg, n, rng, phi=0.5 * math.pi, n_src=None):
    """Positions and velocities for one star: a self-consistent Plummer ball
    (so it does not breathe or evaporate on its own) placed on the inbound leg
    of a bound eccentric orbit, in the y = 0 plane.

    Keeping the orbit in the equatorial plane matters: all the debris then
    inherits that plane, so the disk it forms lands where the ray marcher's
    own equatorial disk lives, and the two read as one structure.

    Dropped part-way down the inbound leg rather than at apocentre.  The orbit
    is the same orbit either way -- same a, same e, same pericentre -- but the
    slow crawl through apocentre is where nearly all of the period goes, and
    starting below it means the stretch and the shredding begin within seconds
    instead of a minute of watching a dot drift.  The radial and tangential
    velocity components come straight from vis-viva plus the orbit's angular
    momentum, so the starting state sits exactly on the intended ellipse."""
    m_star, r_star = cfg["m_star"], cfg["r_star"]
    r_apo, r_peri, gm = cfg["r_apo"], cfg["r_peri"], cfg["gm"]

    # The ball is built cold and centrally concentrated: a dense core survives
    # a little longer than its envelope, which is what puts the kink in the
    # stream and keeps the two tidal tails clearly separate rather than letting
    # the whole thing smear into one even fan.
    p, v = plummer_sphere(n, m_star, r_star * 0.42, 1.0, rng, rcut=2.4)

    a_orb = 0.5 * (r_apo + r_peri)
    ecc = (r_apo - r_peri) / (r_apo + r_peri)
    r0 = float(np.clip(cfg.get("r_start", r_apo), r_peri * 1.6, r_apo))
    v0 = math.sqrt(max(gm * (2.0 / r0 - 1.0 / a_orb), 0.0))
    l_orb = math.sqrt(max(gm * a_orb * (1.0 - ecc * ecc), 0.0))
    v_tan = l_orb / r0
    v_rad = -math.sqrt(max(v0 * v0 - v_tan * v_tan, 0.0))   # inbound

    # placed at azimuth phi and moving so its angular momentum points along -y,
    # the same sense the ray-marched disk turns in, so the disk it eventually
    # forms rotates the right way
    sp, cp = math.sin(phi), math.cos(phi)
    rhat = np.array([sp, 0.0, cp])
    that = np.array([-cp, 0.0, sp])
    p = p + rhat * r0
    v = v + that * v_tan + rhat * v_rad

    # The first n_src are a fair random subset of the same draw -- the sample
    # is i.i.d., so taking a prefix of it is taking a smaller Plummer sphere of
    # the same shape -- and they carry the whole mass between them.  Keeping
    # them contiguous is what lets the integrator take the source set as a
    # range and skip the rest of the O(N^2) inner loop entirely.
    n_src = max(1, min(int(n_src or n), n))
    m = np.zeros(n, dtype=np.float32)
    m[:n_src] = m_star / n_src
    s = np.full(n, cfg["soft"], dtype=np.float32)
    s[:n_src] = cfg.get("soft_src", cfg["soft"])
    return (np.ascontiguousarray(p, dtype=np.float32),
            np.ascontiguousarray(v, dtype=np.float32), m, s)


# ---------------------------------------------------------------------------
# 4) Star cluster collapse -- every particle is massive, full O(N^2)
# ---------------------------------------------------------------------------

def preset_cluster(rng):
    G = 1.0
    n = 4500
    m_tot = 1000.0
    a = 22.0
    P, V = plummer_sphere(n, m_tot, a, G, rng, vscale=0.35)   # cold, so it collapses

    M = np.full(n, m_tot / n)
    S = np.full(n, 0.45)
    u = rng.uniform(0.0, 1.0, n)[:, None]
    blue = np.array([0.60, 0.75, 1.00])[None, :]
    gold = np.array([1.00, 0.72, 0.40])[None, :]
    shade = rng.uniform(0.5, 1.6, n)[:, None]
    C = (blue * (1.0 - u) + gold * u) * shade
    R = np.full(n, 0.30)

    return Scene("Cluster Collapse", P, V, M, S, C, R,
                 gconst=G, dt=0.012, substeps=2,
                 cam_dist=110.0, cam_pitch=0.4, gain=1.1)


PRESETS = {
    1: preset_solar_system,
    2: preset_galaxy_merger,
    3: preset_black_hole,
    4: preset_cluster,
}


# ---------------------------------------------------------------------------
# GLSL
# ---------------------------------------------------------------------------

FULLSCREEN_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Shared helpers: hashes, value noise, a seamless cube-mapped star field and a
# faint nebula.  Injected into every pass that needs to know what the sky looks
# like, because the lensing pass has to evaluate the sky along a *bent* ray.
COMMON_GLSL = """
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), f.x),
                   mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
                   mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float fbm(vec3 x, int oct) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < oct; i++) {
        s += a * vnoise(x);
        x *= 2.02;
        a *= 0.5;
    }
    return s;
}

// Cube-face parameterisation: uniform cells, no pole pinching.
vec2 cubeUV(vec3 d, out float face) {
    vec3 a = abs(d);
    vec2 uv;
    if (a.x >= a.y && a.x >= a.z) { face = d.x > 0.0 ? 0.0 : 1.0; uv = vec2(d.z, d.y) / a.x; }
    else if (a.y >= a.z)          { face = d.y > 0.0 ? 2.0 : 3.0; uv = vec2(d.x, d.z) / a.y; }
    else                          { face = d.z > 0.0 ? 4.0 : 5.0; uv = vec2(d.x, d.y) / a.z; }
    return uv * 0.5 + 0.5;
}

vec3 starLayer(vec2 uv, float face, float scale, float thresh, float bright) {
    vec2 p = uv * scale + face * 71.7;
    vec2 id = floor(p);
    vec2 f = fract(p);
    vec3 acc = vec3(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 o = vec2(float(x), float(y));
            vec2 cid = id + o;
            float h = hash12(cid + face * 17.3);
            if (h < thresh) continue;
            vec2 sp = o + hash22(cid * 1.37 + face);
            float d = length(f - sp);
            float mag = hash12(cid * 2.11 + face * 3.7);
            float core = exp(-d * d * 2600.0) + 0.22 * exp(-d * d * 320.0);
            float tcol = hash12(cid * 5.13 + face * 0.9);
            vec3 tint = mix(vec3(0.72, 0.82, 1.05), vec3(1.05, 0.86, 0.62), tcol);
            acc += tint * core * bright * pow(mag, 6.0);
        }
    }
    return acc;
}

vec3 skyColor(vec3 d) {
    float face;
    vec2 uv = cubeUV(d, face);
    vec3 c = vec3(0.0);
    c += starLayer(uv, face, 40.0, 0.90, 2.2);
    c += starLayer(uv, face, 110.0, 0.86, 0.50);
    c += starLayer(uv, face, 260.0, 0.80, 0.11);

    // faint structured nebulosity + a broad galactic band
    float n = fbm(d * 2.6 + 11.0, 4);
    float n2 = fbm(d * 6.1 - 4.0, 3);
    float band = exp(-pow(abs(dot(d, normalize(vec3(0.22, 1.0, 0.35)))) * 3.1, 2.0));
    vec3 neb = mix(vec3(0.014, 0.018, 0.044), vec3(0.044, 0.018, 0.038), n2);
    c += neb * pow(n, 3.2) * 0.85;
    c += vec3(0.012, 0.014, 0.024) * band * (0.35 + 0.65 * n);
    return c;
}
"""

# Point sprites are drawn into TWO colour attachments: the usual additive HDR
# colour, and a (weight * camera distance, weight) pair.  Dividing one by the
# other in a later pass recovers the brightness-weighted mean distance of
# whatever landed in that texel, which is what lets the lensing pass put a
# particle's light where the bent ray actually crosses it instead of assuming
# every particle sits at infinity.  See LENS_PICKUP_NOTE.
PARTICLE_VS = """
#version 330
in vec3 in_pos;
in vec4 in_attr;          // rgb tint, w = world-space radius
uniform mat4 u_viewProj;
uniform vec3 u_camPos;
uniform vec3 u_bhPos;
uniform float u_pxScale;  // 0.5 * viewport_height / tan(fovy/2)
uniform float u_gain;
uniform float u_nearR;    // world-space radius of the near-field shell
uniform int u_layer;      // 0 = everything, 1 = near field only, 2 = far only
out vec3 v_col;
out float v_fade;
out float v_dist;
void main() {
    if (u_layer != 0) {
        bool isNear = distance(in_pos, u_bhPos) <= u_nearR;
        if ((u_layer == 1) != isNear) {
            // off the far clip plane: the sprite is another layer's business
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            gl_PointSize = 0.0;
            v_col = vec3(0.0);
            v_fade = 0.0;
            v_dist = 0.0;
            return;
        }
    }
    gl_Position = u_viewProj * vec4(in_pos, 1.0);
    float d = max(distance(in_pos, u_camPos), 1e-4);
    float px = in_attr.w * u_pxScale / d;
    // Below one pixel a point sprite stops shrinking, so fold the lost area
    // back into the brightness -- otherwise distant tidal tails read far too
    // bright and the whole field flattens out.
    float fade = clamp(px * px, 0.0, 1.0);
    gl_PointSize = clamp(px * 2.0, 1.0, 96.0);
    v_fade = fade * u_gain * (in_attr.w > 0.0 ? 1.0 : 0.0);
    v_col = in_attr.rgb;
    v_dist = d;
}
"""

PARTICLE_FS = """
#version 330
in vec3 v_col;
in float v_fade;
in float v_dist;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_depth;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    // A wide gaussian, not a tight one: debris is drawn as overlapping soft
    // blobs so a stream reads as a filament instead of a scatter of specks.
    float g = exp(-r2 * 2.6) - 0.0743;      // gaussian core, zero at the rim
    float w = g * v_fade;
    f_color = vec4(v_col * w, w);
    f_depth = vec4(w * v_dist, w, 0.0, 0.0);
}
"""

# --- orbit / mission-path lines: thin flat-colour polylines, drawn straight
# into the HDR scene buffer alongside the particles so they still pass through
# the same lensing/bloom/tonemap chain -----------------------------------------
ORBIT_VS = """
#version 330
in vec3 in_pos;
in vec3 in_col;
uniform mat4 u_viewProj;
uniform vec3 u_camPos;
out vec3 v_col;
out float v_dist;
void main() {
    gl_Position = u_viewProj * vec4(in_pos, 1.0);
    v_col = in_col;
    v_dist = max(distance(in_pos, u_camPos), 1e-4);
}
"""

ORBIT_FS = """
#version 330
in vec3 v_col;
in float v_dist;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_depth;
void main() {
    f_color = vec4(v_col, 1.0);
    f_depth = vec4(v_dist, 1.0, 0.0, 0.0);
}
"""

# --- flat pass: sky + particles, used whenever the lensing pass is off -------
FLAT_FS = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_scene;
uniform sampler2D u_near;
uniform vec3 u_camRight, u_camUp, u_camFwd;
uniform vec3 u_camPos, u_bhPos;
uniform float u_tanHalf, u_aspect;
uniform float u_rs, u_diskIn;
uniform sampler3D u_gas;
uniform vec2 u_gasHalf;
uniform vec3 u_gasOff;
uniform float u_gasBright, u_gasOpacity;
uniform int u_gasMode;
__COMMON__

vec3 diskColor(float t) {
    t = clamp(t, 0.0, 1.6);
    vec3 c;
    if (t < 0.5) c = mix(vec3(0.55, 0.11, 0.02), vec3(1.00, 0.42, 0.06), t / 0.5);
    else if (t < 1.0) c = mix(vec3(1.00, 0.42, 0.06), vec3(1.00, 0.86, 0.54), (t - 0.5) / 0.5);
    else c = mix(vec3(1.00, 0.86, 0.54), vec3(0.86, 0.93, 1.10), (t - 1.0) / 0.6);
    return c;
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(u_camFwd + u_camRight * ndc.x * u_tanHalf * u_aspect
                                 + u_camUp * ndc.y * u_tanHalf);
    vec3 parts = texture(u_scene, v_uv).rgb + texture(u_near, v_uv).rgb;
    vec3 col = skyColor(rd) + parts;

    // With lensing off the light travels in a straight line, but the debris is
    // still a volume and still has to be integrated -- otherwise turning the
    // lensing off makes the gas disappear rather than un-bend.  Entry and exit
    // come from a slab test against the grid, so the march is the same length
    // whatever the camera distance.
    if (u_gasBright > 0.0) {
        vec3 ro = (u_camPos - u_bhPos) / u_rs - u_gasOff;
        vec3 bmax = vec3(u_gasHalf.x, u_gasHalf.y, u_gasHalf.x);
        vec3 inv = 1.0 / rd;
        vec3 ta = (-bmax - ro) * inv, tb = (bmax - ro) * inv;
        vec3 lo = min(ta, tb), hi = max(ta, tb);
        float t0 = max(max(max(lo.x, lo.y), lo.z), 0.0);
        float t1 = min(min(hi.x, hi.y), hi.z);
        if (t1 > t0) {
            float dt = (t1 - t0) / 160.0;
            float trans = 1.0;
            for (int i = 0; i < 160; i++) {
                vec3 q = ro + rd * (t0 + (float(i) + 0.5) * dt);
                vec3 gc = vec3(q.x / u_gasHalf.x, q.y / u_gasHalf.y,
                               q.z / u_gasHalf.x) * 0.5 + 0.5;
                // faded across the outermost cells, exactly as the lensing
                // pass does it: the grid must never show itself as a straight
                // edge ruled across the sky
                vec3 e = smoothstep(vec3(1.0), vec3(0.88), abs(gc * 2.0 - 1.0));
                float g = texture(u_gas, clamp(gc, 0.0, 1.0)).r * e.x * e.y * e.z;
                if (g > 0.004) {
                    vec3 gcol;
                    if (u_gasMode == 1) {
                        gcol = mix(vec3(0.12, 0.16, 0.42), vec3(0.72, 0.34, 0.52),
                                   smoothstep(0.02, 0.30, g));
                        gcol = mix(gcol, vec3(1.00, 0.90, 0.74),
                                   smoothstep(0.28, 0.75, g));
                    } else {
                        gcol = diskColor(
                            pow(clamp(u_diskIn / max(length(q), 1.2), 0.02, 2.0), 0.75)
                            + 0.92 * g);
                    }
                    // Emission and absorption both from the real density, so
                    // this is an ordinary emitting, self-absorbing gas.
                    float rho = g * g;
                    col += trans * gcol * rho * u_gasBright * dt;
                    trans *= exp(-u_gasOpacity * rho * dt);
                }
                if (trans < 0.004) break;
            }
        }
    }
    f_color = vec4(col, 1.0);
}
"""

# --- the black hole ---------------------------------------------------------
GARGANTUA_FS = """
#version 330
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_scene;   // far-field particles: colour
uniform sampler2D u_sceneD;  // far-field particles: (w*dist, w)
uniform sampler2D u_near;    // near-field particles: colour
uniform sampler2D u_nearD;   // near-field particles: (w*dist, w)
uniform mat4 u_viewProj;
uniform vec3 u_camPos;
uniform vec3 u_camRight, u_camUp, u_camFwd;
uniform float u_tanHalf, u_aspect;
uniform vec3 u_bhPos;
uniform float u_rs;          // world units per Schwarzschild radius
uniform float u_time;
uniform float u_diskIn, u_diskOut;
uniform int u_steps;
uniform float u_diskBright;
uniform float u_spin;        // +-1, sense of disk rotation
uniform float u_jetBright;   // 0 = no jet
uniform float u_jetLen;      // how far the beams reach, in r_s
uniform float u_jetRad;      // beam radius at the base
uniform float u_jetTwist;    // strength of the helical striping
uniform float u_nearR;       // near-field shell radius, in r_s
uniform float u_diskH;       // disk scale-height multiplier
uniform sampler3D u_gas;     // N-body debris as a density field
uniform vec2 u_gasHalf;      // its half-extents in r_s: (x and z, y)
uniform vec3 u_gasOff;       // grid centre, relative to the hole, in r_s
uniform float u_gasBright;   // 0 = no gas (particles are being drawn instead)
uniform float u_gasOpacity;
uniform float u_gasStep;     // march step inside the grid: a couple of cells
uniform int u_gasMode;       // 0 = accretion flow, 1 = a cool nebula

__COMMON__

// Offset from the hole, in r_s, to a coordinate in the debris density grid.
vec3 gasCoord(vec3 w) {
    vec3 q = w - u_gasOff;
    return vec3(q.x / u_gasHalf.x, q.y / u_gasHalf.y, q.z / u_gasHalf.x) * 0.5 + 0.5;
}

// Density at a point, faded to nothing across the outermost tenth of the grid.
// Without that fade the box is visible as a straight edge ruled across the sky
// wherever material reaches a face, which is not a thing space does.
float gasAt(vec3 gc) {
    if (any(lessThan(gc, vec3(0.0))) || any(greaterThan(gc, vec3(1.0)))) return 0.0;
    vec3 e = smoothstep(vec3(1.0), vec3(0.88), abs(gc * 2.0 - 1.0));
    return texture(u_gas, gc).r * e.x * e.y * e.z;
}

// Colour of the gas.  Mode 0 is an accretion flow, taking its temperature from
// depth in the potential -- the shock and compression heating that turns cool
// debris into a glowing disk -- plus a density term standing in for a body
// still hot and opaque in its own right.  Without that second term an intact
// star reads as cold as the outermost debris, because it is just as far out,
// and comes out blood red when it should be a warm white.  Mode 1 is a cold
// cloud lit from within: nothing is heating it from a centre, so its colour
// follows density alone.
vec3 gasColor(float g, float r);

// Cheap blackbody-ish ramp, t = 0 (cool outer disk) .. 1+ (inner, doppler boosted)
vec3 diskColor(float t) {
    t = clamp(t, 0.0, 1.6);
    vec3 c;
    if (t < 0.5) c = mix(vec3(0.55, 0.11, 0.02), vec3(1.00, 0.42, 0.06), t / 0.5);
    else if (t < 1.0) c = mix(vec3(1.00, 0.42, 0.06), vec3(1.00, 0.86, 0.54), (t - 0.5) / 0.5);
    else c = mix(vec3(1.00, 0.86, 0.54), vec3(0.86, 0.93, 1.10), (t - 1.0) / 0.6);
    return c;
}

vec3 gasColor(float g, float r) {
    if (u_gasMode == 1) {
        vec3 c = mix(vec3(0.12, 0.16, 0.42), vec3(0.72, 0.34, 0.52),
                     smoothstep(0.02, 0.30, g));
        return mix(c, vec3(1.00, 0.90, 0.74), smoothstep(0.28, 0.75, g));
    }
    float t = pow(clamp(u_diskIn / max(r, 1.2), 0.02, 2.0), 0.75) + 0.92 * g;
    return diskColor(t);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(u_camFwd + u_camRight * ndc.x * u_tanHalf * u_aspect
                                 + u_camUp * ndc.y * u_tanHalf);

    // Work in Schwarzschild radii with the hole at the origin.
    vec3 p = (u_camPos - u_bhPos) / u_rs;
    vec3 v = rd;

    // Null geodesics of the Schwarzschild metric in these coordinates obey
    //     d2x/dl2 = -1.5 * h^2 * x / r^5 ,  h = |x cross v|  (conserved),
    // with r_s = 1.  Integrated below with velocity Verlet.  h only depends on
    // the camera ray, not on where along it we start, so it is safe to compute
    // before the vacuum skip below moves p forward.
    vec3 hvec = cross(p, v);
    float h2 = dot(hvec, hvec);

    // Vacuum skip: far outside the disk, spacetime is essentially flat and the
    // ray travels in a straight line, so jump analytically to where curvature
    // starts to matter instead of spending the march-step budget crossing
    // empty space one small step at a time.  Without this, pulling the camera
    // back a few hundred r_s starves the integrator before it ever reaches the
    // photon sphere and the whole lensing effect silently disappears.
    // How far out there is anything to find.  Never skipped past the
    // near-field shell, because the particle pickup inside the march is the
    // only thing that places that material correctly -- and never past the
    // debris grid either, or a camera pulled back beyond it would see the
    // skip's own sphere as a hard arc cut through the gas.
    float gasReach = 1.05 * max(u_gasHalf.x, u_gasHalf.y) + length(u_gasOff);
    float enterR = max(max(u_diskOut * 3.0, 60.0), u_nearR * 1.05);
    if (u_gasBright > 0.0) enterR = max(enterR, gasReach);
    int steps = u_steps;
    if (dot(p, p) > enterR * enterR) {
        float A = dot(v, v);
        float B = 2.0 * dot(p, v);
        float C = dot(p, p) - enterR * enterR;
        float disc = B * B - 4.0 * A * C;
        float t = disc >= 0.0 ? (-B - sqrt(disc)) / (2.0 * A) : -1.0;
        if (t > 0.0) {
            p += v * t;          // negligible bending accumulated out here
        } else {
            steps = 0;            // path's closest approach never reaches enterR
        }
    }

    float r0 = length(p);
    // marched out past the near-field shell even when the camera is closer in,
    // so debris behind the hole is still picked up on the way out
    float escR = max(max(u_diskOut * 1.6, r0 * 1.15 + 6.0), u_nearR * 1.08);
    // far enough out for the beams to run off the edge of the frame rather
    // than stopping wherever the integration happened to give up
    if (u_jetBright > 0.0) escR = max(escR, u_jetLen * 1.45);
    if (u_gasBright > 0.0) escR = max(escR, gasReach);

    // Far-field particles in FRONT of the hole are still on the straight part
    // of the ray, so they belong exactly where the flat projection drew them.
    // Split front from back by the recorded distance of each texel rather than
    // by which particle went where, so a texel holding both is handled too.
    float camR = distance(u_camPos, u_bhPos);
    vec4 fd = texture(u_sceneD, v_uv);
    float frontMask = 1.0 - smoothstep(camR * 0.96, camR * 1.04,
                                       fd.x / max(fd.y, 1e-5));
    vec3 frontCol = texture(u_scene, v_uv).rgb * frontMask * step(1e-5, fd.y);

    vec3 col = vec3(0.0);
    float trans = 1.0;
    bool captured = false;

    // A fixed fraction of the first step, chosen per pixel.  Marching a smooth
    // emissive volume with a step size that every pixel shares lays down
    // visible contour rings through it; offsetting the phase decorrelates
    // neighbouring pixels and turns those rings into fine noise, which the
    // bloom then smooths away.  Keyed off the pixel and not the clock, so it
    // is stable from frame to frame rather than crawling.
    float jitter = hash12(gl_FragCoord.xy);

    vec3 acc = -1.5 * h2 * p / pow(dot(p, p), 2.5);

    for (int i = 0; i < steps; i++) {
        float r = length(p);
        if (r < 1.0) { captured = true; break; }
        if (r > escR && dot(p, v) > 0.0) break;
        if (trans < 0.004) break;

        // The cap used to be 1.3 everywhere, which is far finer than empty
        // space needs and is what made a long beam or a wide debris field run
        // the step budget out before the ray got there.
        float dt = clamp(0.11 * (r - 0.92), 0.02, 3.5);
        // Enough to resolve the disk's scale height.  The old clamp here drove
        // the step down to 0.035 anywhere near the midplane, because a plane
        // has to be caught exactly; a slab only has to be sampled, and a few
        // samples across its thickness is plenty.
        if (r < u_diskOut * 1.3)
            dt = min(dt, max(0.45 * u_diskH * (0.32 + 0.052 * r), 0.05));
        // inside the beams, step finely enough to resolve their structure
        if (u_jetBright > 0.0 && abs(p.y) < u_jetLen
            && length(p.xz) < u_jetRad * 3.0) dt = min(dt, 0.40);
        // and finely enough not to step over the debris grid's smoothing
        // length, which after three blur passes is a couple of cells.  Scaled
        // to the grid rather than fixed: the grid stretches to reach wherever
        // a star is dropped from, and a step fixed at the size a compact one
        // wants runs the budget out long before the ray gets across a wide one
        vec3 gp = p - u_gasOff;
        if (u_gasBright > 0.0 && abs(gp.y) < u_gasHalf.y
            && max(abs(gp.x), abs(gp.z)) < u_gasHalf.x) dt = min(dt, u_gasStep);
        if (i == 0) dt *= 0.25 + 0.75 * jitter;

        vec3 pPrev = p;
        p += v * dt + 0.5 * acc * dt * dt;
        vec3 accNew = -1.5 * h2 * p / pow(dot(p, p), 2.5);
        v += 0.5 * (acc + accNew) * dt;
        acc = accNew;

        // Sampled at a per-pixel point within the step rather than always at
        // its middle.  Every volume below -- jet, disk, debris -- is sampled
        // here, and sampling them all at the same phase across a whole screen
        // of rays lays down contour rings through them; scattering the phase
        // turns those into fine noise that the bloom then smooths away.
        vec3 mid = mix(pPrev, p, 0.25 + 0.5 * jitter);

        // --- polar jets ---------------------------------------------------
        // Accumulated as a volume along the ray rather than drawn as
        // particles: a beam made of points stays a dotted line no matter how
        // many you throw at it, where an emissive volume actually glows -- and
        // being integrated inside the geodesic march, it gets lensed with
        // everything else.
        //
        // Shaped the way a real jet is shaped rather than as a flashlight
        // cone.  A magnetically collimated beam barely opens out over its
        // whole visible length, carries a hot near-white spine down the axis,
        // and takes its texture from the field wound helically around it plus
        // the knots that advect outward along it.  The wide smooth cone this
        // replaces read as a spotlight; this reads as a beam.
        if (u_jetBright > 0.0) {
            float ay = abs(mid.y);
            if (ay > 0.35 && ay < u_jetLen * 1.35) {
                float h = ay / u_jetLen;
                float cone = u_jetRad * (0.62 + 0.75 * h);   // barely flares
                float rxz = length(mid.xz);
                if (rxz < cone * 2.4) {
                    float q = rxz / cone;

                    // the pattern climbs the beam with time, so the jet
                    // visibly streams outward instead of shimmering in place
                    float turb = fbm(vec3(mid.xz * 1.25,
                                          ay * 0.33 - u_time * 1.30), 4);
                    // twist that unwinds with height: this is what gives the
                    // beam running filaments instead of a smooth wash
                    float tw = 2.0 * atan(mid.z, mid.x) - ay * 0.42 + u_time * 0.55;
                    float helix = 1.0 - u_jetTwist * (0.5 - 0.5 * cos(tw + 5.0 * turb));

                    float spine  = exp(-q * q * 4.0);             // hot axis
                    float sheath = 0.50 * exp(-q * q * 1.15) * helix;
                    float prof = (spine + sheath) * (0.35 + 1.05 * turb);

                    // a bright knot at the launch point, fading fast: the base
                    // of a jet is the brightest part of it
                    prof += 0.55 * spine * exp(-(ay - 1.4) * (ay - 1.4) * 0.10);

                    // Surface brightness falls as the beam widens and cools,
                    // and both ends are ramped rather than cut.  A beam that
                    // simply stops at a radius has a flat end hanging in
                    // space, which is the one thing that reads instantly as a
                    // drawn object rather than as something lit.
                    float ends = smoothstep(0.35, 1.10, ay)
                               * (1.0 - smoothstep(0.72, 1.32, h));
                    float glow = 0.52 * prof * ends * exp(-h * 1.45)
                               / (0.55 + 0.95 * h);
                    vec3 jcol = mix(vec3(0.36, 0.58, 1.30),
                                    vec3(0.96, 0.98, 1.20),
                                    clamp(spine * 1.25, 0.0, 1.0));
                    col += trans * jcol * glow * u_jetBright * dt;
                }
            }
        }

        // --- near-field particles, picked up along the geodesic ------------
        // LENS_PICKUP_NOTE.  The N-body points are drawn flat, by the ordinary
        // projection, into a buffer that also records how far each texel's
        // light was from the camera.  A bent ray then finds a particle by
        // matching both screen position and distance: every point in space has
        // a unique (pixel, distance) pair, so that buffer is a usable stand-in
        // for the particle field itself, and the light gets deposited where
        // the ray really crosses it.
        //
        // This is what stops the orbiting debris being lensed into a different
        // place from the disk it belongs to.  Sampling the buffer by the
        // escaping ray's direction instead -- the obvious thing, and what was
        // here before -- quietly assumes every particle is infinitely far
        // away, which is badly wrong for material a few r_s from the hole and
        // throws its image off into an arc of its own.
        if (length(mid) < u_nearR) {
            vec3 wmid = u_bhPos + mid * u_rs;
            vec4 pc = u_viewProj * vec4(wmid, 1.0);
            if (pc.w > 1e-5) {
                vec2 uv = pc.xy / pc.w * 0.5 + 0.5;
                if (all(greaterThanEqual(uv, vec2(0.0))) &&
                    all(lessThanEqual(uv, vec2(1.0)))) {
                    vec4 nd = texture(u_nearD, uv);
                    if (nd.y > 1e-5) {
                        float dp = nd.x / nd.y;
                        float d0 = distance(u_camPos, u_bhPos + pPrev * u_rs);
                        float d1 = distance(u_camPos, u_bhPos + p * u_rs);
                        // half-open, so consecutive steps tile the ray exactly
                        // once -- except where the ray turns back on itself,
                        // and a second pickup there is a second image
                        if (dp >= min(d0, d1) && dp < max(d0, d1)) {
                            col += trans * texture(u_near, uv).rgb;
                        }
                    }
                }
            }
        }

        // --- the disk, as a slab with thickness --------------------------
        // It used to be a mathematical plane, caught by testing whether the
        // step crossed y = 0.  That is why it looked paper-thin and searingly
        // bright edge-on: a plane has no edge to see, so all of its light
        // arrived in a single sample and collapsed onto one line of pixels.
        // Now it is integrated as a volume with a scale height that flares
        // outward, so from the side it is a band of glowing gas of some
        // definite depth, and the light is spread through that depth.  The
        // 1/H below keeps the face-on brightness the same as the old surface
        // model, so only the edge-on view changes.
        {
            vec3 x = mid;
            float rr = length(x.xz);
            float hh = u_diskH * (0.32 + 0.052 * rr);
            float vprof = exp(-(x.y * x.y) / (hh * hh));
            if (rr > u_diskIn && rr < u_diskOut && vprof > 0.004) {
                float tn = (rr - u_diskIn) / (u_diskOut - u_diskIn);

                // Differential rotation: the noise field is sampled in the
                // frame co-rotating with the local Keplerian angular velocity,
                // so the lanes shear the way real disk material does.
                //
                // Winding it straight off the clock does not work for long.
                // The inner disk laps the outer one, the pattern is dragged
                // into ever tighter spirals, and within a few hundred time
                // units those spirals are finer than a pixel and alias into
                // concentric rings.  So the shear is run on two half-cycle
                // offset copies and cross-faded between them: each copy is
                // never advected more than half a period before it is reset
                // under cover of the other, which keeps the winding bounded
                // while the disk still visibly turns.
                float per = 45.0;
                float ta = fract(u_time / per);
                float tb = fract(u_time / per + 0.5);
                float om = u_spin * 0.7071 * pow(rr, -1.5);
                float wa = 1.0 - abs(2.0 * ta - 1.0);
                vec3 nz = vec3(0.0, 0.0, log(rr) * 2.2);
                float dens = 0.0;
                for (int k = 0; k < 2; k++) {
                    float dphi = om * ((k == 0 ? ta : tb) - 0.5) * per;
                    float cs = cos(dphi), sn = sin(dphi);
                    vec2 q = vec2(cs * x.x - sn * x.z, sn * x.x + cs * x.z);
                    nz.xy = q * 0.42;
                    dens += fbm(nz, 3) * (k == 0 ? wa : 1.0 - wa);
                }
                dens = pow(clamp(dens * 1.7 - 0.32, 0.0, 1.0), 1.25);
                float lanes = 0.40 + 0.60 * dens;

                float radial = smoothstep(0.0, 0.10, tn) * (1.0 - smoothstep(0.45, 1.0, tn));
                float emis = pow(u_diskIn / rr, 2.1);

                // orbital velocity of the emitting gas (c = 1, r_s = 1 -> GM = 1/2)
                vec3 vel = normalize(cross(vec3(0.0, 1.0, 0.0), x)) * u_spin * sqrt(0.5 / rr);
                vec3 nobs = -normalize(v);
                float b2 = clamp(dot(vel, vel), 0.0, 0.98);
                float gam = inversesqrt(1.0 - b2);
                float dop = 1.0 / max(gam * (1.0 - dot(vel, nobs)), 0.05);
                float grav = sqrt(max(1.0 - 1.0 / rr, 0.02));
                float shift = clamp(dop * grav, 0.05, 3.2);

                float boost = shift * shift * shift;
                float temp = pow(u_diskIn / rr, 0.75) * shift;
                vec3 emit = diskColor(temp) * emis * lanes * radial * boost * u_diskBright;

                float seg = vprof * dt / (1.772 * hh);
                col += trans * emit * seg;
                // opacity has to fade out with brightness too, or a disk turned
                // down to nothing still casts a dark band across the lensed sky
                trans *= exp(-2.4 * dens * radial * seg
                             * clamp(u_diskBright, 0.0, 1.0));
            }
        }

        // --- the N-body debris, as gas -----------------------------------
        // The same medium everywhere: the star still falling in, the stream it
        // is drawn out into, and the ring it settles into are one density
        // field with one emission model, so they run into each other instead
        // of meeting at a seam.  Temperature comes from depth in the potential
        // and uses the disk's own colour ramp, which is what makes the debris
        // and the drawn disk read as the same substance.
        if (u_gasBright > 0.0) {
            {
                float g = gasAt(gasCoord(mid));
                if (g > 0.004) {
                    // Emission and absorption both from the real density, so
                    // this is an ordinary emitting, self-absorbing gas: thick
                    // material glows at a fixed surface brightness and thin
                    // material is fainter in proportion to how much of it the
                    // ray passes through.  Taking emission from the encoded
                    // square root instead is tempting, because it flatters the
                    // faint stuff -- but a ray crossing a hundred r_s of
                    // near-vacuum then accumulates as much light as one
                    // crossing the disk, and the whole frame fogs over.
                    float rho = g * g;
                    col += trans * gasColor(g, length(mid - u_gasOff))
                                 * rho * u_gasBright * dt;
                    trans *= exp(-u_gasOpacity * rho * dt);
                }
            }
        }
    }

    if (!captured) {
        vec3 dir = normalize(v);
        vec3 sky = skyColor(dir);

        // Re-project the escaping ray into the particle buffer: the N-body
        // points are effectively at infinity relative to the hole, so the bent
        // direction is the right thing to look up -- that is what makes the
        // simulated disk and the star field smear into Einstein arcs.
        vec4 clip = u_viewProj * vec4(dir, 0.0);
        if (clip.w > 1e-5) {
            vec2 uv = clamp(clip.xy / clip.w * 0.5 + 0.5, 0.0, 1.0);
            vec2 e = smoothstep(vec2(0.0), vec2(0.012), uv)
                   * (1.0 - smoothstep(vec2(0.988), vec2(1.0), uv));
            // only the material genuinely behind the hole: those points are
            // far enough away that the escaping direction is the right lookup,
            // which is the one case the at-infinity assumption actually holds
            vec4 bd = texture(u_sceneD, uv);
            float backMask = smoothstep(camR * 0.96, camR * 1.04,
                                        bd.x / max(bd.y, 1e-5));
            sky += texture(u_scene, uv).rgb * e.x * e.y * backMask;
        }
        col += trans * sky;
    }

    // added last and unattenuated: this layer is nearer than everything above
    col += frontCol;

    f_color = vec4(col, 1.0);
}
"""

BRIGHT_FS = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_thresh;
void main() {
    // 4-tap box downsample first: sampling a quarter-res target with a single
    // fetch turns bright cores into blocky squares once they are blurred.
    vec3 c = texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb
           + texture(u_src, v_uv + u_texel * vec2( 1.0, -1.0)).rgb
           + texture(u_src, v_uv + u_texel * vec2(-1.0,  1.0)).rgb
           + texture(u_src, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
    c *= 0.25;
    float l = max(max(c.r, c.g), c.b);
    float k = max(l - u_thresh, 0.0) / max(l, 1e-4);
    f_color = vec4(c * k, 1.0);
}
"""

BLUR_FS = """
#version 330
in vec2 v_uv;
// Two outputs because this also blurs the particle buffer, which carries a
// companion distance attachment; that attachment must be written to something
// defined, and zero is what "no particle here" means to everything that reads
// it.  When the target has only one attachment the second write is discarded.
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_aux;
uniform sampler2D u_src;
uniform vec2 u_dir;          // texel-sized step along x or y
void main() {
    // 9-tap gaussian folded into 5 linearly-interpolated samples
    const float o[3] = float[3](0.0, 1.3846153846, 3.2307692308);
    const float w[3] = float[3](0.2270270270, 0.3162162162, 0.0702702703);
    vec3 c = texture(u_src, v_uv).rgb * w[0];
    for (int i = 1; i < 3; i++) {
        c += texture(u_src, v_uv + u_dir * o[i]).rgb * w[i];
        c += texture(u_src, v_uv - u_dir * o[i]).rgb * w[i];
    }
    f_color = vec4(c, 1.0);
    f_aux = vec4(0.0);
}
"""

COMPOSITE_FS = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_bloomAmt;
uniform float u_exposure;
uniform float u_time;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = texture(u_hdr, v_uv).rgb;
    c += texture(u_bloom, v_uv).rgb * u_bloomAmt;
    c = aces(c * u_exposure);
    c = pow(c, vec3(1.0 / 2.2));

    vec2 d = v_uv - 0.5;
    c *= 1.0 - 0.85 * dot(d, d) * dot(d, d) * 4.0;

    // ordered-ish dither keeps the deep background free of banding
    float n = fract(sin(dot(v_uv * 1024.0 + u_time, vec2(12.9898, 78.233))) * 43758.5453);
    c += (n - 0.5) / 255.0;
    f_color = vec4(c, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Matrix helpers (row-major maths convention; transposed on upload)
# ---------------------------------------------------------------------------

def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-9)
    upv = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = upv
    m[2, :3] = -fwd
    m[0, 3] = -np.dot(right, eye)
    m[1, 3] = -np.dot(upv, eye)
    m[2, 3] = np.dot(fwd, eye)
    return m, right, upv, fwd


def setu(prog, name, value):
    """Set a uniform if the program actually declares it (dead uniforms get
    optimised out by the driver, which would otherwise raise)."""
    try:
        u = prog[name]
    except KeyError:
        return
    if isinstance(value, np.ndarray) and value.ndim == 2:
        u.write(np.ascontiguousarray(value.T, dtype="f4").tobytes())
    else:
        u.value = value


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class Camera:
    def __init__(self):
        self.target = np.zeros(3, dtype=np.float64)
        self.dist = 100.0
        self.yaw = 0.0
        self.pitch = 0.4
        self.fov = 55.0
        self.home = None
        self.min_dist = 0.05

    def adopt(self, scene):
        self.target = scene.cam_target.astype(np.float64).copy()
        self.dist = scene.cam_dist
        self.yaw = scene.cam_yaw
        self.pitch = scene.cam_pitch
        self.fov = scene.fov
        self.home = (self.target.copy(), self.dist, self.yaw, self.pitch, self.fov)
        # A camera that flies past the photon sphere sees nothing but black --
        # not a bug, but indistinguishable from one at a glance.  Floor the
        # distance just outside it so the hole always stays legible.
        self.min_dist = 2.6 * scene.world_rs if scene.bh else 0.05

    def reset(self):
        if self.home is None:
            return
        target, dist, yaw, pitch, fov = self.home
        self.target = target.copy()
        self.dist, self.yaw, self.pitch, self.fov = dist, yaw, pitch, fov

    @property
    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array([cp * math.sin(self.yaw),
                                                   math.sin(self.pitch),
                                                   cp * math.cos(self.yaw)])

    def orbit(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch = max(-1.52, min(1.52, self.pitch + dpitch))

    def dolly(self, factor):
        self.dist = max(self.min_dist, min(self.dist * factor, 1.0e5))

    def pan(self, dx, dy):
        """Slide the orbit target across the view plane. The step scales with
        distance so panning feels the same whether you are looking at a solar
        system or at a pair of colliding galaxies."""
        _, right, up, _ = look_at(self.eye, self.target)
        k = self.dist * 0.0016 * math.tan(math.radians(self.fov) * 0.5)
        self.target = self.target + right * (-dx * k) + up * (dy * k)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, ctx, win_size, scale):
        self.ctx = ctx
        self.scale = float(scale)
        self.win_w, self.win_h = win_size

        # live tunables driven by the GUI
        self.exposure = 0.95
        self.bloom_amt = 0.55
        self.bloom_thresh = 0.75
        self.disk_bright = 2.1
        self.particle_gain = 1.0
        self.steps = ARGS.steps
        self.spin = -1.0          # matches the sense the N-body disk orbits in
        self.jet_bright = 0.0     # volumetric polar jets, driven by the App
        self.jet_len = 70.0
        self.jet_rad = 2.40       # collimated: the beam barely opens out
        self.jet_twist = 0.70     # depth of the helical striping along it
        # Radius of the shell inside which particles are lensed properly, by
        # being picked up along the bent ray instead of assumed to be at
        # infinity.  Fixed rather than tracking the disk, so that particles
        # never swap layers mid-flight and pop.  See LENS_PICKUP_NOTE.
        self.near_radius = 70.0   # in r_s
        self.disk_h = 1.0         # scale-height multiplier for the drawn disk
        self.gas_bright = 2.20    # emission per unit density of the debris gas
        self.gas_opacity = 0.70   # how much of itself it hides behind
        self.gas_soft = 5.0       # blur radius, in texels, for "screen" gas
        self.gas_on = True

        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.quad_vbo = ctx.buffer(quad.tobytes())

        def frag(src):
            return src.replace("__COMMON__", COMMON_GLSL)

        self.prog_particle = ctx.program(vertex_shader=PARTICLE_VS,
                                         fragment_shader=PARTICLE_FS)
        self.prog_orbit = ctx.program(vertex_shader=ORBIT_VS, fragment_shader=ORBIT_FS)
        self.prog_flat = self._fs(frag(FLAT_FS))
        self.prog_lens = self._fs(frag(GARGANTUA_FS))
        self.prog_bright = self._fs(BRIGHT_FS)
        self.prog_blur = self._fs(BLUR_FS)
        self.prog_comp = self._fs(COMPOSITE_FS)

        # The debris density field.  A byte per cell, sampled as a float: see
        # the gas section up top for why it is quantised.
        self.gas_tex = ctx.texture3d((GAS_NX, GAS_NY, GAS_NZ), 1, dtype="f1")
        self.gas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.gas_tex.repeat_x = False
        self.gas_tex.repeat_y = False
        self.gas_tex.repeat_z = False
        self.clear_gas()

        self.pos_vbo = ctx.buffer(reserve=MAX_N * 12, dynamic=True)
        self.attr_vbo = ctx.buffer(reserve=MAX_N * 16, dynamic=True)
        self.particle_vao = ctx.vertex_array(
            self.prog_particle,
            [(self.pos_vbo, "3f", "in_pos"), (self.attr_vbo, "4f", "in_attr")])

        # Orbit / mission-path lines: one contiguous pos+colour buffer holding
        # every polyline the current scene has, drawn as a handful of separate
        # LINE_STRIPs (self.line_ranges: [(first, count, kind, name), ...]) so
        # any subset -- all orbits, one mission -- can be shown independently.
        self.orbit_pos_vbo = None
        self.orbit_col_vbo = None
        self.orbit_vao = None
        self.line_ranges = []

        self._targets = []
        self._build_targets()

    def _build_targets(self):
        self.rw = max(320, int(self.win_w * self.scale))
        self.rh = max(240, int(self.win_h * self.scale))
        self.bw = max(64, self.rw // 4)
        self.bh_ = max(64, self.rh // 4)
        for obj in self._targets:
            obj.release()
        # Particles land in two layers, each with a companion distance buffer:
        # "near" is everything inside the shell where light visibly bends and
        # is picked up along the geodesic, "far" is everything outside it.
        self.tex_scene, self.tex_sceneD, self.fbo_scene = self._layer(self.rw, self.rh)
        self.tex_near, self.tex_nearD, self.fbo_near = self._layer(self.rw, self.rh)
        self.tex_hdr, self.fbo_hdr = self._target(self.rw, self.rh)
        # scratch for the screen-space gas blur, which runs before pass 2 and
        # so cannot borrow the HDR target
        self.tex_blur, self.fbo_blur = self._target(self.rw, self.rh)
        self.tex_b0, self.fbo_b0 = self._target(self.bw, self.bh_)
        self.tex_b1, self.fbo_b1 = self._target(self.bw, self.bh_)
        self._targets = [self.fbo_scene, self.fbo_near, self.fbo_hdr,
                         self.fbo_blur, self.fbo_b0, self.fbo_b1,
                         self.tex_scene, self.tex_sceneD, self.tex_near,
                         self.tex_nearD, self.tex_hdr, self.tex_blur,
                         self.tex_b0, self.tex_b1]

    def resize(self, w, h, scale=None):
        if scale is not None:
            self.scale = float(scale)
        self.win_w, self.win_h = int(w), int(h)
        self._build_targets()

    def _fs(self, source):
        return self.ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=source)

    def _tex(self, w, h, filt=moderngl.LINEAR):
        tex = self.ctx.texture((w, h), 4, dtype="f2")
        tex.filter = (filt, filt)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _target(self, w, h):
        tex = self._tex(w, h)
        return tex, self.ctx.framebuffer(color_attachments=[tex])

    def _layer(self, w, h):
        """A particle layer: additive colour plus a (weight * distance,
        weight) buffer.  The distance buffer is point-sampled -- interpolating
        between two texels holding unrelated distances invents a depth that
        nothing in the scene is at, and the lensing pass would then deposit
        light at it."""
        col = self._tex(w, h)
        dist = self._tex(w, h, moderngl.NEAREST)
        return col, dist, self.ctx.framebuffer(color_attachments=[col, dist])

    def _blit(self, prog):
        vao = self.ctx.vertex_array(prog, [(self.quad_vbo, "2f", "in_pos")])
        vao.render(moderngl.TRIANGLES, vertices=3)
        vao.release()

    def upload_gas(self, arr):
        self.gas_tex.write(arr.tobytes())

    def clear_gas(self):
        self.gas_tex.write(bytes(GAS_NX * GAS_NY * GAS_NZ))

    def upload_scene(self, scene):
        self.attr_vbo.write(scene.attrib.tobytes())
        self._upload_polylines(scene)

    def update_attrib(self, scene, lo, hi):
        """Re-upload just one slice of the colour/size buffer (used to re-tint
        tidal debris as it falls deeper in) rather than the whole array."""
        chunk = np.ascontiguousarray(scene.attrib[lo:hi], dtype=np.float32)
        self.attr_vbo.write(chunk.tobytes(), offset=lo * 16)

    def _upload_polylines(self, scene):
        if self.orbit_vao is not None:
            self.orbit_vao.release()
            self.orbit_pos_vbo.release()
            self.orbit_col_vbo.release()
            self.orbit_vao = None
        self.line_ranges = []
        if not scene.polylines:
            return
        pos_chunks, col_chunks = [], []
        first = 0
        for pl in scene.polylines:
            p = np.ascontiguousarray(pl["pos"], dtype=np.float32)
            n = p.shape[0]
            c = np.tile(np.asarray(pl["color"], dtype=np.float32), (n, 1))
            pos_chunks.append(p)
            col_chunks.append(c)
            self.line_ranges.append((first, n, pl["kind"], pl["name"]))
            first += n
        pos_all = np.concatenate(pos_chunks)
        col_all = np.concatenate(col_chunks)
        self.orbit_pos_vbo = self.ctx.buffer(pos_all.tobytes())
        self.orbit_col_vbo = self.ctx.buffer(col_all.tobytes())
        self.orbit_vao = self.ctx.vertex_array(
            self.prog_orbit,
            [(self.orbit_pos_vbo, "3f", "in_pos"), (self.orbit_col_vbo, "3f", "in_col")])

    def draw(self, scene, cam, positions, sim_time, lensing,
             vertex_count=None, line_kinds=frozenset(), extras=()):
        ctx = self.ctx
        aspect = self.rw / float(self.rh)
        view, right, up, fwd = look_at(cam.eye, cam.target)
        near = max(cam.dist * 1e-3, 1e-3)
        far = cam.dist * 60.0 + 5000.0
        proj = perspective(cam.fov, aspect, near, far)
        vp = proj @ view
        eye = cam.eye.astype(np.float32)
        tan_half = math.tan(math.radians(cam.fov) * 0.5)

        # --- pass 1: particles into the HDR scene buffers ---------------------
        # Split into a far and a near layer whenever the lensing pass will run,
        # because the two are placed on screen by completely different means:
        # the near layer is picked up along the bent ray, the far layer is
        # composited either straight through (in front of the hole) or by the
        # escaping direction (behind it).  See LENS_PICKUP_NOTE in the shader.
        split = bool(lensing and scene.bh)
        near_r = self.near_radius * scene.world_rs
        bh_pos = (float(positions[0, 0]), float(positions[0, 1]),
                  float(positions[0, 2])) if len(positions) else (0.0, 0.0, 0.0)
        n_draw = vertex_count or scene.n

        self.pos_vbo.write(positions.tobytes())
        self.fbo_scene.use()
        ctx.viewport = (0, 0, self.rw, self.rh)
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        p = self.prog_particle
        setu(p, "u_viewProj", vp)
        setu(p, "u_camPos", tuple(float(x) for x in eye))
        setu(p, "u_bhPos", bh_pos)
        setu(p, "u_pxScale", 0.5 * self.rh / tan_half)
        setu(p, "u_gain", scene.gain * self.particle_gain)
        setu(p, "u_nearR", near_r)
        setu(p, "u_layer", 2 if split else 0)
        self.particle_vao.render(moderngl.POINTS, vertices=n_draw)
        for first, count in extras:
            self.particle_vao.render(moderngl.POINTS, vertices=count, first=first)

        # --- orbit / mission-path lines, same buffer, same blend mode --------
        if line_kinds and self.orbit_vao is not None:
            setu(self.prog_orbit, "u_viewProj", vp)
            setu(self.prog_orbit, "u_camPos", tuple(float(x) for x in eye))
            for first, count, kind, name in self.line_ranges:
                if (kind == "orbit" and "orbit" in line_kinds) or \
                   (kind == "mission" and name in line_kinds):
                    self.orbit_vao.render(moderngl.LINE_STRIP, vertices=count, first=first)

        ctx.disable(moderngl.BLEND)

        # --- pass 1a': screen-space gas ------------------------------------
        # Two separable blurs over the particle buffer.  A galaxy's light adds
        # up along the ray with nothing absorbing it, so spreading each star
        # over a few pixels and letting them overlap is not a stand-in for a
        # gas -- it is what an optically thin one does.  The blur conserves
        # total light, so this changes how the material looks without changing
        # how much of it there is.
        if scene.gas == "screen" and self.gas_on and self.gas_soft > 0.01:
            # Split into whatever number of passes keeps each one's taps close
            # enough together to cover the ground between them.  The kernel is
            # five samples wide at one-texel spacing; stretched much past that
            # it stops being a blur and starts being a regular pattern of
            # samples, which shows up as a grid laid over the whole galaxy.
            passes = max(1, int(math.ceil(self.gas_soft / 1.25)))
            k = self.gas_soft / passes
            for _ in range(passes):
                self.fbo_blur.use()
                ctx.viewport = (0, 0, self.rw, self.rh)
                self.tex_scene.use(0)
                setu(self.prog_blur, "u_src", 0)
                setu(self.prog_blur, "u_dir", (k / self.rw, 0.0))
                self._blit(self.prog_blur)

                self.fbo_scene.use()
                ctx.viewport = (0, 0, self.rw, self.rh)
                self.tex_blur.use(0)
                setu(self.prog_blur, "u_dir", (0.0, k / self.rh))
                self._blit(self.prog_blur)

        # --- pass 1b: the near layer, on its own so the lensing pass can
        # place it along the geodesic instead of at infinity -----------------
        self.fbo_near.use()
        ctx.viewport = (0, 0, self.rw, self.rh)
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        if split:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            setu(p, "u_layer", 1)
            self.particle_vao.render(moderngl.POINTS, vertices=n_draw)
            for first, count in extras:
                self.particle_vao.render(moderngl.POINTS, vertices=count, first=first)
            ctx.disable(moderngl.BLEND)

        # --- pass 2: lensing (or straight-through) ---------------------------
        self.fbo_hdr.use()
        ctx.viewport = (0, 0, self.rw, self.rh)
        self.tex_scene.use(0)
        self.tex_sceneD.use(1)
        self.tex_near.use(2)
        self.tex_nearD.use(3)
        self.gas_tex.use(4)
        gas_bright = (self.gas_bright
                      if (scene.gas == "volume" and self.gas_on) else 0.0)
        # the grid is centred on the origin, so in the marcher's hole-centred
        # frame it sits at minus the hole's own offset from there
        gas_off = tuple(-float(x) / max(scene.world_rs, 1e-6) for x in bh_pos)
        gas_mode = 0 if scene.bh else 1
        if split:
            g = self.prog_lens
            setu(g, "u_scene", 0)
            setu(g, "u_sceneD", 1)
            setu(g, "u_near", 2)
            setu(g, "u_nearD", 3)
            setu(g, "u_nearR", self.near_radius)
            setu(g, "u_viewProj", vp)
            setu(g, "u_camPos", tuple(float(x) for x in eye))
            setu(g, "u_camRight", tuple(float(x) for x in right))
            setu(g, "u_camUp", tuple(float(x) for x in up))
            setu(g, "u_camFwd", tuple(float(x) for x in fwd))
            setu(g, "u_tanHalf", tan_half)
            setu(g, "u_aspect", aspect)
            setu(g, "u_bhPos", bh_pos)
            setu(g, "u_rs", scene.world_rs)
            setu(g, "u_time", sim_time)
            setu(g, "u_diskIn", scene.disk_in)
            setu(g, "u_diskOut", scene.disk_out)
            setu(g, "u_steps", self.steps)
            setu(g, "u_diskBright", self.disk_bright)
            setu(g, "u_spin", self.spin)
            setu(g, "u_jetBright", self.jet_bright)
            setu(g, "u_jetLen", self.jet_len)
            setu(g, "u_jetRad", self.jet_rad)
            setu(g, "u_jetTwist", self.jet_twist)
            setu(g, "u_diskH", self.disk_h)
            setu(g, "u_gas", 4)
            setu(g, "u_gasHalf", scene.gas_half)
            setu(g, "u_gasOff", gas_off)
            setu(g, "u_gasMode", gas_mode)
            setu(g, "u_gasBright", gas_bright)
            setu(g, "u_gasOpacity", self.gas_opacity)
            setu(g, "u_gasStep", max(1.0, 2.8 * scene.gas_half[0] / GAS_NX))
            self._blit(g)
        else:
            g = self.prog_flat
            setu(g, "u_scene", 0)
            setu(g, "u_near", 2)
            setu(g, "u_camRight", tuple(float(x) for x in right))
            setu(g, "u_camUp", tuple(float(x) for x in up))
            setu(g, "u_camFwd", tuple(float(x) for x in fwd))
            setu(g, "u_camPos", tuple(float(x) for x in eye))
            setu(g, "u_bhPos", bh_pos)
            setu(g, "u_rs", scene.world_rs)
            setu(g, "u_diskIn", scene.disk_in)
            setu(g, "u_tanHalf", tan_half)
            setu(g, "u_aspect", aspect)
            setu(g, "u_gas", 4)
            setu(g, "u_gasHalf", scene.gas_half)
            setu(g, "u_gasOff", gas_off)
            setu(g, "u_gasMode", gas_mode)
            setu(g, "u_gasBright", gas_bright)
            setu(g, "u_gasOpacity", self.gas_opacity)
            self._blit(g)

        # --- pass 3: bloom ---------------------------------------------------
        self.fbo_b0.use()
        ctx.viewport = (0, 0, self.bw, self.bh_)
        self.tex_hdr.use(0)
        setu(self.prog_bright, "u_src", 0)
        setu(self.prog_bright, "u_texel", (0.5 / self.rw, 0.5 / self.rh))
        setu(self.prog_bright, "u_thresh", self.bloom_thresh)
        self._blit(self.prog_bright)

        for _ in range(3):
            self.fbo_b1.use()
            self.tex_b0.use(0)
            setu(self.prog_blur, "u_src", 0)
            setu(self.prog_blur, "u_dir", (1.4 / self.bw, 0.0))
            self._blit(self.prog_blur)

            self.fbo_b0.use()
            self.tex_b1.use(0)
            setu(self.prog_blur, "u_src", 0)
            setu(self.prog_blur, "u_dir", (0.0, 1.4 / self.bh_))
            self._blit(self.prog_blur)

        # --- pass 4: composite to the default framebuffer --------------------
        ctx.screen.use()
        ctx.viewport = (0, 0, self.win_w, self.win_h)
        self.tex_hdr.use(0)
        self.tex_b0.use(1)
        setu(self.prog_comp, "u_hdr", 0)
        setu(self.prog_comp, "u_bloom", 1)
        setu(self.prog_comp, "u_bloomAmt", self.bloom_amt)
        setu(self.prog_comp, "u_exposure", self.exposure)
        setu(self.prog_comp, "u_time", sim_time)
        self._blit(self.prog_comp)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self):
        self.scene = None
        self.key = ARGS.preset
        self.time = 0.0
        self.speed = 1.0
        self.paused = False
        self.rng = np.random.default_rng()
        # tidal-disruption controls
        self.self_gravity = True
        self.viscosity = 0.0025     # circularisation rate, 1 / time unit -- low
                                    # enough that the debris stream stays a
                                    # visible stream for a good while before it
                                    # settles into the disk
        self.inflow = 0.055         # fraction of that which removes L: sets how
                                    # fast material spirals in. With sustain on this
                                    # is throughput, not loss -- what reaches the
                                    # hole is resupplied, so the disk is a steady
                                    # state rather than a dwindling one.
        self.visc_radius = 42.0     # only inside here, where the stream piles up
        self.sustain_disk = True    # resupply accreted material, so the disk persists
        self.feed_radius = 26.0     # where a sustained disk is resupplied
        self.return_radius = 210.0  # past here, scattered material is brought back
        self.star_intact = False    # self-gravity only matters while a star is whole
        self.jet = True             # twin polar jets off the inner disk
        self.jet_rate = 0.0018      # per-frame chance a disk particle is launched
        self.jet_range = 70.0       # jet material is returned to the disk past here
        self.jet_source = 18.0      # jets are fed from inside this radius
        self.jet_base = 2.2         # launch height above the hole
        self.jet_speed = 0.88       # launch speed (c = 1), above local escape
        self.jet_spread = 0.032     # opening angle as a fraction of jet speed
        self.circularising = False  # switched on once the star reaches pericentre

    def load(self, key):
        self.key = key
        self.scene = PRESETS[key](self.rng)
        s = self.scene
        if s.n > MAX_N:
            raise RuntimeError(f"scene {s.name} needs {s.n} particles, MAX_N is {MAX_N}")
        if s.n_sat > MAX_SAT:
            raise RuntimeError(f"scene {s.name} needs {s.n_sat} satellites, MAX_SAT is {MAX_SAT}")
        k_upload(s.n, s.pos, s.vel, s.mass, s.soft)
        if s.kill_drift:
            k_com_drift(s.n_dynamic, s.n_src)
        if s.n_sat > 0:
            k_upload_satellites(s.n_sat, s.sat_parent, s.sat_u, s.sat_v,
                                s.sat_radius, s.sat_rate, s.sat_phase0)
            k_update_satellites(s.n_dynamic, s.n_sat, 0.0)
        lo, hi = self.src_bounds()
        k_accel(s.n_dynamic, lo, hi, s.gconst, s.pw_rs)
        self.time = 0.0
        self.circularising = False
        self.star_intact = False
        return s

    def src_bounds(self):
        """Which particles act as gravity sources this step, as a half-open
        range.  The hole at index 0 is handled separately by k_accel through
        the Paczynski-Wiita term, so it is deliberately left out of the range
        whenever a pseudo-Newtonian hole is present."""
        s = self.scene
        if s.star_n:
            # Only the star that is still whole self-gravitates.  Once it is a
            # stretched stream the hole dominates utterly, so dropping the
            # older slots costs nothing physically and keeps the inner loop at
            # one star's worth of sources no matter how many have been dropped.
            if self.self_gravity and self.star_intact and s.star_lo >= 1:
                return s.star_lo, s.star_lo + (s.star_src or s.star_n)
            return 1, 1
        lo = 1 if s.pw_rs > 0.0 else 0
        return lo, s.n_src

    def spawn_star(self, azimuth=0.5 * math.pi):
        """Drop a fresh star onto the hole.  Slots are reused oldest-first once
        they run out, so you can keep feeding it."""
        s = self.scene
        if not s.star_n:
            return False
        slot = s.star_spawned % s.star_slots
        lo = 1 + slot * s.star_n
        p, v, m, soft = make_star(s.star_cfg, s.star_n, self.rng, azimuth,
                                  n_src=s.star_src)
        k_write_block(lo, s.star_n, p, v, m, soft)
        s.star_spawned += 1
        # grow the live range to cover every slot used so far; every dynamic
        # particle here is also a gravity source, so the star holds itself
        # together and its debris keeps pulling on itself
        live = 1 + min(s.star_spawned, s.star_slots) * s.star_n
        s.n_dynamic = max(s.n_dynamic, live)
        s.n_base = s.n_dynamic + s.n_sat
        self.star_intact = True
        s.star_lo = lo
        # Stretch the density grid out to wherever this star is coming from.
        # Left at its default the star would spend its whole infall outside
        # the grid and simply not be drawn, and the grid's own edge would show
        # up as a straight line across the sky the moment debris reached it.
        reach = max(s.star_cfg.get("r_start", s.star_cfg["r_apo"]),
                    0.75 * s.star_cfg["r_apo"])
        half = min(max(1.15 * reach, 110.0), 280.0)
        s.gas_half = (half, min(max(0.22 * half, 30.0), 75.0))
        # and material is only "scattered away" past where the star came from,
        # or a star dropped from far out is teleported into the disk the
        # instant it appears
        self.return_radius = max(self.return_radius, 1.4 * reach)
        k_balance_momentum(s.n_dynamic)
        src_lo, src_hi = self.src_bounds()
        s.n_src = src_hi - src_lo + 1
        k_accel(s.n_dynamic, src_lo, src_hi, s.gconst, s.pw_rs)
        return True

    def step(self):
        s = self.scene
        dt = s.dt * self.speed
        # Self-gravity is only what holds a star together on the way in; once
        # it is a spread-out stream the hole dominates utterly, so the source
        # range collapses to the hole afterwards and the cost drops from
        # O(N^2) to O(N).  See Sim.src_bounds.
        src_lo, src_hi = self.src_bounds()
        s.n_src = src_hi - src_lo + (1 if s.pw_rs > 0.0 else 0)
        eating = bool(s.star_n) and s.n_dynamic > 1
        for _ in range(s.substeps):
            k_kick(s.n_dynamic, 0.5 * dt)
            k_drift(s.n_dynamic, dt)
            k_accel(s.n_dynamic, src_lo, src_hi, s.gconst, s.pw_rs)
            k_kick(s.n_dynamic, 0.5 * dt)
            self.time += dt
            if eating:
                s.accreted += k_swallow(1, s.n_dynamic, s.pw_rs,
                                        1 if self.sustain_disk else 0,
                                        s.gconst * s.mass[0], self.feed_radius,
                                        self.return_radius, self.jet_range)
        if s.n_sat > 0:
            k_update_satellites(s.n_dynamic, s.n_sat, self.time)
        if eating:
            # Neither the viscous damping nor the resupply teleport conserves
            # momentum, and with a star heavy enough to matter that leak is
            # enough to walk the hole clean out of its own disk over a few
            # thousand time units -- which then scatters the disk. Pinning the
            # total momentum to zero every frame keeps the hole where the disk
            # is; with momentum conserved this is exactly what the hole's
            # velocity would have been anyway.
            k_balance_momentum(s.n_dynamic)
            k_recentre(s.n_dynamic)
            if self.viscosity > 0.0 and self.circularising:
                frac = 1.0 - math.exp(-self.viscosity * dt * s.substeps)
                k_circularise(1, s.n_dynamic, frac, self.inflow,
                              self.visc_radius, s.pw_rs)
            if self.jet and self.circularising:
                k_jet(1, s.n_dynamic, self.jet_rate, self.jet_source,
                      self.jet_speed, self.jet_spread, self.jet_base, s.pw_rs)
        if s.recycle is not None:
            lo, hi = s.recycle
            k_recycle(lo, hi, s.gconst * s.mass[0], s.pw_rs,
                      s.disk_in, s.disk_out, 0.25)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _reset_scissor():
    """Dear ImGui enables GL_SCISSOR_TEST and never restores it, so without this
    the next frame gets clipped to the last panel it drew.

    The error queue is drained first because PyOpenGL validates the *previous*
    call's error flag, and ModernGL leaves stale flags behind (a screenshot
    readback is one way to get one)."""
    while gl.glGetError() != gl.GL_NO_ERROR:
        pass
    gl.glDisable(gl.GL_SCISSOR_TEST)


def _save_shot(ctx, w, h, path):
    from PIL import Image
    data = ctx.screen.read(components=3, alignment=1)
    img = Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)
    print("saved", path)


# ---------------------------------------------------------------------------
# Application state + Dear ImGui panels
# ---------------------------------------------------------------------------

_SCENE_KEYS = {
    pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
    pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4,
}


class App:
    def __init__(self, ctx, renderer, cam, sim):
        self.ctx = ctx
        self.renderer = renderer
        self.cam = cam
        self.sim = sim
        self.scene = None
        self.lensing = True
        self.running = True
        self.fullscreen = ARGS.fullscreen
        self.fps = 0.0
        self.frame_ms = 0.0
        self.pending_scale = ARGS.scale
        self.show_gui = True
        self.show_labels = False
        self.show_orbits = False
        self.extras_on = {}    # name -> bool, populated fresh per scene
        self.mission_on = {}   # mission name -> bool, populated fresh per scene
        self.auto_disk = True  # let the ray-marched disk grow in from the debris
        self.disk_peak = 3.0   # brightness the emergent disk builds up to
        self.jet_peak = 0.22   # brightness the polar jets build up to
        self.star_sprite = 0.300  # point radius inside a star that is still whole
        self.star_glow = 0.0      # 1 while it is a star, 0 once it is a stream
        self.star_bound = 0.0     # fraction of it still inside 2 r_star
        self.star_core = np.zeros(3, dtype=np.float32)
        self.tde_tick = 0         # phase of the every-fourth-frame debris survey
        self.tgt_disk = (0.0, 3.0, 26.0)   # held between those surveys
        self.gas_amp = 1.0        # how much material one particle stands for
        self.debris_inner = 0.0
        self.debris_outer = 0.0
        self.debris_frac = 0.0

    def load(self, key):
        self.scene = self.sim.load(key)
        self.renderer.upload_scene(self.scene)
        self.cam.adopt(self.scene)
        self.mission_on = {pl["name"]: False for pl in self.scene.polylines if pl["kind"] == "mission"}
        self.extras_on = {name: False for name, _, _ in self.scene.extras}
        self.debris_inner = self.debris_outer = self.debris_frac = 0.0
        self.star_glow = self.star_bound = 0.0
        self.tgt_disk = (0.0, self.scene.disk_in, self.scene.disk_out)
        self.renderer.clear_gas()
        if self.scene.star_n:
            # a bare hole: nothing to light up until a star has been torn apart
            self.renderer.disk_bright = 0.0
            self.renderer.jet_bright = 0.0
            self.scene.disk_in, self.scene.disk_out = 3.0, 26.0
        return self.scene

    def update_gas(self):
        """Re-splat the debris into the density grid the renderer marches."""
        s, r = self.scene, self.renderer
        if not (s.gas == "volume" and r.gas_on) or s.n_dynamic <= s.gas_lo:
            return
        r.upload_gas(build_gas(s, self.gas_amp))

    def update_tde(self, positions):
        """Per-frame tidal-disruption bookkeeping: watch the debris, start
        circularisation once the star has actually reached pericentre, fade the
        ray-marched disk in as material piles up, and re-tint the debris by how
        deep in the potential it sits."""
        s, sim, r = self.scene, self.sim, self.renderer
        if not s.star_n or s.star_spawned == 0 or s.n_dynamic <= 1:
            return
        deb = positions[1:s.n_dynamic]

        # Everything below that only feeds a smoothed scalar -- how much of the
        # debris is on a disk-like orbit, where its edges are -- is measured on
        # a stride through the particles rather than all of them.  Tens of
        # thousands of points is a count chosen so a stream draws smoothly; a
        # few thousand is already far more than these statistics need, and at
        # 60 Hz the difference is milliseconds a frame.
        step = max(1, (s.n_dynamic - 1) // 8000)
        sub = deb[::step]
        r_sub = np.linalg.norm(sub, axis=1)
        alive = r_sub < 1.0e3
        n_alive = int(alive.sum())
        if n_alive == 0:
            self.debris_frac = 0.0
            return

        r_alive = r_sub[alive]
        if not sim.circularising and r_alive.min() < 1.35 * s.star_cfg["r_peri"]:
            # pericentre reached, so whatever came off the star is now a stream
            # that wants the viscous damping turned on
            sim.circularising = True

        # "Disk-like" means on a near-circular orbit, not merely nearby: an
        # intact star coasting through apocentre has little radial motion too,
        # and must not be mistaken for a disk.  Comparing its angular momentum
        # against the circular value at the same radius separates the two.
        #
        # Measured a few times a second rather than every frame: it needs the
        # velocities, and pulling those back off the GPU is the most expensive
        # thing in this function.  Nothing it feeds moves quickly -- they are
        # all eased towards over seconds -- so the targets are simply held
        # between measurements, and the easing below still runs every frame.
        self.tde_tick = (self.tde_tick + 1) % 4
        if self.tde_tick == 0:
            # Is the star still a star?  This used to be assumed rather than
            # measured -- self-gravity was switched off the moment the star
            # reached pericentre, on the grounds that by then it is a stream.
            # For the default star it is.  For a dense one on a wide pericentre
            # it is not, and switching its gravity off was what tore it apart:
            # the panel would say it survives the pass and it would come apart
            # anyway.  So ask the particles instead.
            lo = s.star_lo
            if lo >= 1 and lo + s.star_n <= s.n_dynamic:
                blk = positions[lo:lo + s.star_n:step]
                self.star_core = np.median(blk, axis=0)
                d_core = np.linalg.norm(blk - self.star_core, axis=1)
                self.star_bound = float(
                    (d_core < 1.5 * s.star_cfg["r_star"]).mean())
            else:
                self.star_bound = 0.0
            sim.star_intact = self.star_bound > 0.50

            vel_all = vel.to_numpy()[1:s.n_dynamic:step][alive]
            p_alive = sub[alive]
            rhat = p_alive / r_alive[:, None]
            v_r = np.abs(np.sum(vel_all * rhat, axis=1))
            v_c = np.sqrt(0.5 * r_alive) / np.maximum(r_alive - s.pw_rs, 1e-3)
            l_mag = np.linalg.norm(np.cross(p_alive, vel_all), axis=1)
            kappa = l_mag / np.maximum(r_alive * v_c, 1e-9)
            disky = ((r_alive < 60.0) & (v_r < 0.30 * v_c)
                     & (kappa > 0.75) & (kappa < 1.3))
            cnt = int(disky.sum())
            self.debris_frac = cnt / float(n_alive)

            self.tgt_disk = (0.0, s.disk_in, s.disk_out)
            if cnt > 60:
                rr = r_alive[disky]
                # 70th percentile, not 90th: a long-lived disk always has some
                # scattered material way out past the body of it, and letting
                # that set the outer edge inflates the drawn disk until it
                # swallows the frame and flattens the temperature gradient
                self.debris_inner = float(np.percentile(rr, 8))
                self.debris_outer = float(np.percentile(rr, 70))
                # Pinned near the ISCO rather than following the debris.
                # Inside the ISCO material plunges in a few orbits, so the
                # particles there are always sparse -- but a real disk still
                # radiates right down to it, and letting the drawn inner edge
                # drift out to where the particles thin out leaves an obvious
                # empty ring around the hole.
                tgt_in = 3.2
                self.tgt_disk = (self.disk_peak * min(1.0, self.debris_frac / 0.35),
                                 tgt_in,
                                 min(max(self.debris_outer, tgt_in + 3.0), 36.0))

        tgt_bright, tgt_in, tgt_out = self.tgt_disk
        if self.auto_disk:
            # ease toward the target so the disk grows in smoothly instead of
            # snapping around as debris sloshes through pericentre
            k = 0.02
            r.disk_bright += (tgt_bright - r.disk_bright) * k
            s.disk_in += (tgt_in - s.disk_in) * k
            s.disk_out += (tgt_out - s.disk_out) * k
        # the jets are powered by accretion, so they light up with the disk
        tgt_jet = self.jet_peak if (sim.jet and sim.circularising) else 0.0
        tgt_jet *= min(1.0, self.debris_frac / 0.35)
        r.jet_bright += (tgt_jet - r.jet_bright) * 0.02

        self.star_glow += (self.star_bound - self.star_glow) * 0.02
        if s.gas == "volume" and r.gas_on:
            return      # nothing is drawn as a sprite, so nothing to tint

        rad = np.linalg.norm(deb, axis=1)
        # Re-tint by depth in the potential: a star still on its way in stays
        # stellar warm-white, debris shock-heats and brightens as it spirals in.
        # Written channel by channel rather than by broadcasting a pair of
        # colours: this one runs over every particle every frame, and the
        # broadcast form allocates half a dozen arrays that size to do it.
        t = np.clip((rad - 3.0) * (1.0 / 34.0), 0.0, 1.0)
        boost = 0.85 + 1.10 * (1.0 - t) ** 2
        col = np.empty((rad.size, 3), dtype=np.float32)
        col[:, 0] = boost                             # 1.00 hot and cool alike
        col[:, 1] = (0.88 - 0.18 * t) * boost
        col[:, 2] = (0.74 - 0.32 * t) * boost
        # anything well off the disk plane is in a jet: give the beams their own
        # hot blue-white so they separate from the disk instead of reading as
        # stray debris
        jetting = np.abs(deb[:, 1]) > 0.45 * np.maximum(rad, 1e-6)
        col[jetting] = np.array([0.72, 0.86, 1.25], dtype=np.float32)
        # Deliberately fat and faint rather than small and bright.  Fourteen
        # thousand points cannot cover a stream a hundred r_s long, so drawn
        # sharp they read as a scatter of separate specks; drawn as broad soft
        # blobs at a fraction of the brightness they overlap into something
        # continuous, which is what the stream actually is.
        size = np.where(jetting, np.float32(0.110), np.float32(0.130))

        # A star is a filled body, a stream is a thread, and one point radius
        # cannot draw both: at the size that makes a hundred-r_s stream look
        # like a fine filament, a whole star is a faint spray of dots.  So the
        # sprite radius follows each particle's distance from the still-bound
        # core -- fat and overlapping inside it, fine out in the tails -- and
        # the core is brightened as well, because a star should read as the
        # brightest thing in frame until the hole takes it apart.  Applied only
        # to the star most recently dropped; anything older is long since a
        # stream, and its sprites stay thin.
        lo = s.star_lo
        if self.star_glow > 0.01 and lo >= 1 and lo + s.star_n <= s.n_dynamic:
            blk = positions[lo:lo + s.star_n]
            d = np.linalg.norm(blk - self.star_core, axis=1)
            w = self.star_glow * np.exp(-(d / (1.6 * s.star_cfg["r_star"])) ** 2)
            j = slice(lo - 1, lo - 1 + s.star_n)
            size[j] = size[j] + (self.star_sprite - size[j]) * w
            col[j] *= (1.0 + 1.2 * w)[:, None]

        s.attrib[1:s.n_dynamic, 0:3] = col
        s.attrib[1:s.n_dynamic, 3] = size
        r.update_attrib(s, 1, s.n_dynamic)

    def spawn_star(self):
        """Drop the star on the far side of the hole from the camera, nudged off
        the shadow.  Spawned at a fixed azimuth it usually lands outside a
        42-degree field of view entirely, so you press the key and see nothing;
        from back there it is centred in frame and falls in past the hole."""
        return self.sim.spawn_star(self.cam.yaw + math.pi + 0.44)

    def extra_ranges(self):
        """(first, count) for each switched-on block of static extras."""
        return [(lo, n) for name, lo, n in self.scene.extras
                if self.extras_on.get(name)]

    def visible_lines(self):
        """The set draw() checks each polyline against: the literal string
        "orbit" gates every orbit-kind line at once, one flag each mission."""
        lines = set(name for name, on in self.mission_on.items() if on)
        if self.show_orbits:
            lines.add("orbit")
        return lines


# Every key the main loop binds, in one place, so the panel that lists them
# cannot drift away from the loop that handles them.
KEY_HELP = [
    ("1 2 3 4", "load a scene"),
    ("R", "restart the current scene"),
    ("space", "pause / resume"),
    ("[ ]", "slower / faster  (also - and +)"),
    ("X", "drop a star on the black hole"),
    ("V", "draw the material as gas, or as points"),
    ("L", "gravitational lensing on / off"),
    ("N", "body name labels"),
    ("O", "orbit paths"),
    ("F", "reset the camera"),
    ("G", "hide / show these panels"),
    ("F11", "fullscreen"),
    ("esc", "quit"),
    ("W S", "move in / out"),
    ("A D", "swing left / right"),
    ("Q E", "swing up / down"),
    ("arrows", "pan"),
]


SCENE_NAMES = {
    1: "1  Solar System disk",
    2: "2  Galaxy Merger",
    3: "3  Black hole  (10 M_sun)",
    4: "4  Star Cluster Collapse",
}


def draw_labels(app, positions):
    """Planet/dwarf-planet name tags and mission end-point markers, drawn as a
    Dear ImGui overlay -- projected through the same view/projection matrices
    the 3D pass uses, so they track the camera exactly without needing any
    dedicated 3D text-rendering machinery."""
    scene, cam, r = app.scene, app.cam, app.renderer
    aspect = r.win_w / float(r.win_h)
    view, _, _, _ = look_at(cam.eye, cam.target)
    near = max(cam.dist * 1e-3, 1e-3)
    far = cam.dist * 60.0 + 5000.0
    vp = perspective(cam.fov, aspect, near, far) @ view
    dl = imgui.get_foreground_draw_list()

    def project(p):
        clip = vp @ np.array([p[0], p[1], p[2], 1.0])
        if clip[3] <= 1e-4:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        sx = (ndc[0] * 0.5 + 0.5) * r.win_w
        sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * r.win_h
        if sx < -200 or sx > r.win_w + 200 or sy < -200 or sy > r.win_h + 200:
            return None
        return sx, sy

    if app.show_labels:
        col = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.88))
        for idx, name in scene.labels:
            xy = project(positions[idx])
            if xy is None:
                continue
            dl.add_circle_filled(imgui.ImVec2(xy[0], xy[1]), 2.5, col)
            dl.add_text(imgui.ImVec2(xy[0] + 7, xy[1] - 7), col, name)

    for pl in scene.polylines:
        if pl["kind"] != "mission" or not app.mission_on.get(pl["name"], False):
            continue
        xy = project(pl["pos"][-1])
        if xy is None:
            continue
        # the 3D line colour is deliberately dim (so it survives ACES tonemapping
        # without blowing out to white); brighten it back up for this flat,
        # untonemapped UI marker, which needs no such restraint.
        boost = 1.9
        mc = tuple(min(ch * boost, 1.0) for ch in pl["color"])
        col = imgui.get_color_u32(imgui.ImVec4(mc[0], mc[1], mc[2], 1.0))
        dl.add_circle_filled(imgui.ImVec2(xy[0], xy[1]), 3.5, col)
        dl.add_text(imgui.ImVec2(xy[0] + 8, xy[1] - 7), col, pl["name"])


def draw_gui(app):
    """Dear ImGui control panels. Every widget drives live state -- nothing here
    is cosmetic."""
    imgui.set_next_window_pos(imgui.ImVec2(12, 12), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(348, 336), imgui.Cond_.first_use_ever)
    imgui.begin("Simulation")
    imgui.push_item_width(-122)
    sim, scene, r = app.sim, app.scene, app.renderer

    if imgui.button("Pause" if not sim.paused else "Resume", imgui.ImVec2(96, 0)):
        sim.paused = not sim.paused
    imgui.same_line()
    if imgui.button("Restart", imgui.ImVec2(96, 0)):
        app.load(sim.key)
    imgui.same_line()
    if imgui.button("Step", imgui.ImVec2(96, 0)):
        sim.step()

    changed, val = imgui.slider_float("speed", sim.speed, 0.05, 16.0, "%.2fx",
                                      imgui.SliderFlags_.logarithmic)
    if changed:
        sim.speed = val
    changed, val = imgui.slider_int("substeps", scene.substeps, 1, 8)
    if changed:
        scene.substeps = val
    changed, val = imgui.slider_float("dt", scene.dt, 0.002, 0.4, "%.4f",
                                      imgui.SliderFlags_.logarithmic)
    if changed:
        scene.dt = val

    imgui.separator_text("state")
    imgui.text(f"scene       {scene.name}")
    imgui.text(f"sim time    {sim.time:10.2f}")
    imgui.text(f"particles   {scene.n:10d}")
    imgui.text(f"gravitating {scene.n_src:10d}")
    if scene.n_sat:
        imgui.text(f"satellites  {scene.n_sat:10d}")
    imgui.text(f"pair terms  {scene.n_dynamic * scene.n_src * scene.substeps / 1e6:8.1f} M/frame")
    imgui.pop_item_width()
    imgui.end()

    # --- scenes ------------------------------------------------------------
    imgui.set_next_window_pos(imgui.ImVec2(12, 360), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(348, 150), imgui.Cond_.first_use_ever)
    imgui.begin("Scenes")
    for key in (1, 2, 3, 4):
        if imgui.radio_button(SCENE_NAMES[key], sim.key == key) and sim.key != key:
            app.load(key)
    imgui.end()

    # --- camera ------------------------------------------------------------
    imgui.set_next_window_pos(imgui.ImVec2(12, 522), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(348, 228), imgui.Cond_.first_use_ever)
    imgui.begin("Camera")
    imgui.push_item_width(-92)
    cam = app.cam
    changed, val = imgui.slider_float("distance", cam.dist, cam.min_dist, 8000.0, "%.1f",
                                      imgui.SliderFlags_.logarithmic)
    if changed:
        cam.dist = val
    changed, val = imgui.slider_float("yaw", math.degrees(cam.yaw), -180.0, 180.0, "%.0f deg")
    if changed:
        cam.yaw = math.radians(val)
    changed, val = imgui.slider_float("pitch", math.degrees(cam.pitch), -87.0, 87.0, "%.0f deg")
    if changed:
        cam.pitch = math.radians(val)
    changed, val = imgui.slider_float("fov", cam.fov, 15.0, 100.0, "%.0f deg")
    if changed:
        cam.fov = val
    t = cam.target
    imgui.text(f"target  {t[0]:8.1f} {t[1]:8.1f} {t[2]:8.1f}")
    if imgui.button("Reset view", imgui.ImVec2(120, 0)):
        cam.reset()
    imgui.same_line()
    if imgui.button("Centre target", imgui.ImVec2(150, 0)):
        cam.target[:] = 0.0
    imgui.text("drag LMB orbit | MMB or shift+LMB pan")
    imgui.text("wheel zoom | arrows pan | WASDQE fly")
    if imgui.collapsing_header("keyboard"):
        for key, what in KEY_HELP:
            imgui.text(f"{key:<10s} {what}")
    imgui.pop_item_width()
    imgui.end()

    # --- black hole --------------------------------------------------------
    w = app.renderer.win_w
    if scene.bh:
        imgui.set_next_window_pos(imgui.ImVec2(w - 386, 12), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(374, 276), imgui.Cond_.first_use_ever)
        imgui.begin("Black Hole (ray marched)")
        imgui.push_item_width(-132)
        changed, val = imgui.checkbox("gravitational lensing", app.lensing)
        if changed:
            app.lensing = val
        changed, val = imgui.slider_int("march steps", r.steps, 40, 640)
        if changed:
            r.steps = val
        changed, val = imgui.slider_float("lensed shell", r.near_radius, 12.0, 150.0, "%.0f r_s")
        if changed:
            r.near_radius = val
        if imgui.is_item_hovered():
            imgui.set_tooltip("Particles inside this radius are lensed by being picked "
                              "up along the bent ray, so their images land in the same "
                              "place as the disk they belong to. Outside it they are far "
                              "enough away to composite straight through. Raising it "
                              "costs march steps.")
        changed, val = imgui.slider_float("disk brightness", r.disk_bright, 0.0, 6.0, "%.2f")
        if changed:
            r.disk_bright = val
        changed, val = imgui.slider_float("disk inner", scene.disk_in, 1.2, 12.0, "%.2f r_s")
        if changed:
            scene.disk_in = min(val, scene.disk_out - 1.0)
        changed, val = imgui.slider_float("disk outer", scene.disk_out, 6.0, 70.0, "%.1f r_s")
        if changed:
            scene.disk_out = max(val, scene.disk_in + 1.0)
        changed, val = imgui.slider_float("disk thickness", r.disk_h, 0.25, 3.0, "%.2f")
        if changed:
            r.disk_h = val
        changed, val = imgui.checkbox("flip disk spin", r.spin > 0.0)
        if changed:
            r.spin = 1.0 if val else -1.0
        imgui.separator_text("geometry")
        imgui.text(f"mass         {BH_SOLAR_MASSES:.0f} M_sun")
        imgui.text(f"horizon      1.00 r_s  ({BH_RS_KM:6.1f} km)")
        imgui.text(f"photon ring  1.50 r_s  ({1.5 * BH_RS_KM:6.1f} km)")
        imgui.text(f"ISCO         3.00 r_s  ({3.0 * BH_RS_KM:6.1f} km)")
        imgui.text(f"camera       {cam.dist / max(scene.world_rs, 1e-6):8.2f} r_s")
        imgui.pop_item_width()
        imgui.end()

    if scene.star_n:
        # --- tidal disruption ----------------------------------------------------
        imgui.set_next_window_pos(imgui.ImVec2(360, 12), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(336, 610), imgui.Cond_.first_use_ever)
        imgui.begin("Tidal Disruption")
        imgui.push_item_width(-118)
        if imgui.button("Spawn star  (X)", imgui.ImVec2(-1, 0)):
            app.spawn_star()
        cfg = scene.star_cfg or {"m_star": 0.0, "r_star": 0.0, "r_peri": 0.0, "r_apo": 0.0}
        changed, val = imgui.slider_float("star mass", cfg["m_star"] * SIM_MASS_TO_SOLAR,
                                          0.05, 4.0, "%.2f M_sun")
        if changed:
            cfg["m_star"] = val / SIM_MASS_TO_SOLAR
        changed, val = imgui.slider_float("star radius", cfg["r_star"], 0.5, 18.0, "%.2f r_s")
        if changed:
            cfg["r_star"] = val
        changed, val = imgui.slider_float("pericentre", cfg["r_peri"], 3.5, 45.0, "%.1f r_s")
        if changed:
            cfg["r_peri"] = min(val, cfg["r_apo"] - 5.0)
        changed, val = imgui.slider_float("apocentre", cfg["r_apo"], 20.0, 320.0, "%.0f r_s")
        if changed:
            cfg["r_apo"] = max(val, cfg["r_peri"] + 5.0)
            # keep the drop point on the inbound leg of whatever orbit this is now
            cfg["r_start"] = max(0.64 * cfg["r_apo"], cfg["r_peri"] * 1.6)
        changed, val = imgui.slider_float("drop at", cfg.get("r_start", cfg["r_apo"]),
                                          10.0, 320.0, "%.0f r_s")
        if changed:
            cfg["r_start"] = min(max(val, cfg["r_peri"] * 1.6), cfg["r_apo"])
        changed, val = imgui.slider_float("star sprite", app.star_sprite, 0.03, 0.70, "%.3f r_s")
        if changed:
            app.star_sprite = val

        if cfg["m_star"] > 0.0:
            # r_t = r_h (M_bh / M_star)^(1/3), with r_h = 1.3 a and a = r_star / 2
            r_t = 0.65 * cfg["r_star"] * (0.5 / cfg["m_star"]) ** (1.0 / 3.0)
            beta = r_t / max(cfg["r_peri"], 1e-6)
            verdict = "full disruption" if beta >= 1.0 else "survives the pass"
            col = imgui.ImVec4(0.5, 1.0, 0.6, 1.0) if beta >= 1.0 else imgui.ImVec4(1.0, 0.8, 0.4, 1.0)
            imgui.text(f"tidal radius {r_t:6.1f} r_s")
            imgui.text_colored(col, f"beta = r_t/r_p = {beta:4.2f}  {verdict}")
            # The star is drawn bigger than the hole, which is the right way round,
            # but it is nothing like the true ratio -- so say so rather than let
            # the picture imply otherwise.  See the preset's docstring.
            r_km = cfg["r_star"] * BH_RS_KM
            imgui.text(f"star radius {r_km:8.0f} km = {r_km / 696340.0:.4f} R_sun")
            imgui.text_colored(imgui.ImVec4(0.65, 0.7, 0.8, 1.0),
                               "(compact for its mass, so the disruption")
            imgui.text_colored(imgui.ImVec4(0.65, 0.7, 0.8, 1.0),
                               " happens where the lensing is visible)")

        imgui.separator_text("debris")
        changed, val = imgui.checkbox("sustain disk (companion feed)", sim.sustain_disk)
        if changed:
            sim.sustain_disk = val
        if sim.sustain_disk:
            changed, val = imgui.slider_float("feed radius", sim.feed_radius, 8.0, 45.0, "%.1f r_s")
            if changed:
                sim.feed_radius = val
        changed, val = imgui.checkbox("self-gravity", sim.self_gravity)
        if changed:
            sim.self_gravity = val
        changed, val = imgui.checkbox("disk grows from debris", app.auto_disk)
        if changed:
            app.auto_disk = val
        changed, val = imgui.slider_float("viscosity", sim.viscosity, 0.0, 0.03, "%.4f")
        if changed:
            sim.viscosity = val
        changed, val = imgui.slider_float("accretion rate", sim.inflow, 0.0, 0.10, "%.3f")
        if changed:
            sim.inflow = val
        imgui.separator_text("polar jets")
        changed, val = imgui.checkbox("twin jets", sim.jet)
        if changed:
            sim.jet = val
        imgui.begin_disabled(not sim.jet)
        changed, val = imgui.slider_float("jet power", app.jet_peak, 0.0, 1.2, "%.2f")
        if changed:
            app.jet_peak = val
        changed, val = imgui.slider_float("jet reach", app.renderer.jet_len, 15.0, 160.0, "%.0f r_s")
        if changed:
            app.renderer.jet_len = val
            sim.jet_range = val
        changed, val = imgui.slider_float("jet width", app.renderer.jet_rad, 0.25, 4.0, "%.2f r_s")
        if changed:
            app.renderer.jet_rad = val
        changed, val = imgui.slider_float("jet twist", app.renderer.jet_twist, 0.0, 1.0, "%.2f")
        if changed:
            app.renderer.jet_twist = val
        changed, val = imgui.slider_float("beam spread", sim.jet_spread, 0.0, 0.20, "%.3f")
        if changed:
            sim.jet_spread = val
        imgui.end_disabled()
        imgui.text(f"stars dropped  {scene.star_spawned:6d}")
        imgui.text(f"{'recycled' if sim.sustain_disk else 'accreted':<14s} {scene.accreted:6d}")
        imgui.text(f"bound fraction {app.debris_frac * 100.0:5.1f} %")
        if app.debris_outer > 0.0:
            imgui.text(f"debris  {app.debris_inner:5.1f} - {app.debris_outer:5.1f} r_s")
        imgui.text("circularising" if sim.circularising else "waiting for pericentre")
        imgui.pop_item_width()
        imgui.end()

    # --- solar system overlays ----------------------------------------------
    if scene.extras or scene.labels:
        imgui.set_next_window_pos(imgui.ImVec2(360, 12), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(336, 300), imgui.Cond_.first_use_ever)
        imgui.begin("Solar System")
        changed, val = imgui.checkbox("body names  (N)", app.show_labels)
        if changed:
            app.show_labels = val
        changed, val = imgui.checkbox("orbit paths  (O)", app.show_orbits)
        if changed:
            app.show_orbits = val
        for name, _, count in scene.extras:
            changed, val = imgui.checkbox(name, app.extras_on.get(name, False))
            if changed:
                app.extras_on[name] = val
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"{count} points, off by default. It sits far outside"
                    " the planets, so seeing it means pulling the camera"
                    " back until they are a knot in the middle.")
        if app.mission_on:
            imgui.separator_text("space mission paths")
            for name in app.mission_on:
                changed, val = imgui.checkbox(name, app.mission_on[name])
                if changed:
                    app.mission_on[name] = val
                    if val and name.startswith("Voyager"):
                        # the paths only mean something next to the boundary they cross
                        for ex in app.extras_on:
                            if ex.startswith("heliosphere"):
                                app.extras_on[ex] = True
        imgui.end()

    # --- render ------------------------------------------------------------
    imgui.set_next_window_pos(imgui.ImVec2(w - 386, 300), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(374, 244), imgui.Cond_.first_use_ever)
    imgui.begin("Render")
    imgui.push_item_width(-132)
    changed, val = imgui.slider_float("exposure", r.exposure, 0.1, 4.0, "%.2f")
    if changed:
        r.exposure = val
    changed, val = imgui.slider_float("bloom", r.bloom_amt, 0.0, 2.0, "%.2f")
    if changed:
        r.bloom_amt = val
    changed, val = imgui.slider_float("bloom threshold", r.bloom_thresh, 0.05, 3.0, "%.2f")
    if changed:
        r.bloom_thresh = val
    changed, val = imgui.slider_float("particle gain", r.particle_gain, 0.1, 4.0, "%.2f")
    if changed:
        r.particle_gain = val
    # Only the scene whose gas has anything worth adjusting gets the panel.
    # The black hole's is on by default and has nothing to tune that the
    # exposure and bloom above do not already cover, so it lives on the V key
    # alone rather than taking up room in every session.
    # Only the scene whose gas has something worth adjusting gets a panel for
    # it.  The black hole's is on by default and needs no dials the exposure
    # and bloom above do not already cover, so it lives on the V key alone
    # rather than taking up room in every session.
    if scene.gas == "screen":
        imgui.separator_text("gas")
        changed, val = imgui.checkbox("draw as gas  (V)", r.gas_on)
        if changed:
            r.gas_on = val
        if imgui.is_item_hovered():
            imgui.set_tooltip("Draw the stars as one continuous glow rather "
                              "than as separate points. A galaxy is optically "
                              "thin, so its light simply adds up along the "
                              "ray, which is what this does.")
        imgui.begin_disabled(not r.gas_on)
        changed, val = imgui.slider_float("softening", r.gas_soft, 0.0, 8.0, "%.2f px")
        if changed:
            r.gas_soft = val
        imgui.end_disabled()
        imgui.separator_text("")
    imgui.push_item_width(-186)
    changed, val = imgui.slider_float("scale", app.pending_scale, 0.35, 1.0, "%.2f")
    if changed:
        app.pending_scale = val
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button("Apply"):
        r.resize(r.win_w, r.win_h, app.pending_scale)
    imgui.text(f"internal {r.rw} x {r.rh}")
    if imgui.button("Toggle fullscreen (F11)", imgui.ImVec2(230, 0)):
        app.toggle_fullscreen()
    imgui.pop_item_width()
    imgui.end()

    # --- performance -------------------------------------------------------
    imgui.set_next_window_pos(imgui.ImVec2(w - 386, 556), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size(imgui.ImVec2(374, 122), imgui.Cond_.first_use_ever)
    imgui.begin("Performance")
    imgui.text(f"{app.fps:6.1f} fps      {app.frame_ms:6.2f} ms")
    imgui.progress_bar(min(app.fps / 120.0, 1.0), imgui.ImVec2(-1, 0), f"{app.fps:.0f} / 120")
    imgui.text(f"taichi arch  {ARGS.arch}")
    imgui.text(f"window       {r.win_w} x {r.win_h}")
    imgui.end()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                    pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)

    info = pygame.display.Info()
    if ARGS.fullscreen:
        w, h = info.current_w, info.current_h
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
    else:
        w = ARGS.width or min(1600, info.current_w - 80)
        h = ARGS.height or min(900, info.current_h - 120)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    try:
        pygame.display.set_mode((w, h), flags, vsync=ARGS.vsync)
    except pygame.error:
        pygame.display.set_mode((w, h), flags)
    pygame.display.set_caption("N-Body Universe + Ray-Marched Black Hole")

    ctx = moderngl.create_context()
    renderer = Renderer(ctx, (w, h), ARGS.scale)
    cam = Camera()
    app = App(ctx, renderer, cam, Sim())

    imgui.create_context()
    io = imgui.get_io()
    io.display_size = (w, h)
    io.set_ini_filename("space_sim_layout.ini")
    imgui.style_colors_dark()
    style = imgui.get_style()
    style.window_rounding = 6.0
    style.frame_rounding = 3.0
    style.window_border_size = 0.0
    style.alpha = 0.94
    impl = PygameRenderer()

    def toggle_fullscreen():
        pygame.display.toggle_fullscreen()
        surf = pygame.display.get_surface()
        nw, nh = surf.get_size()
        io.display_size = (nw, nh)
        renderer.resize(nw, nh)
        app.fullscreen = not app.fullscreen

    app.toggle_fullscreen = toggle_fullscreen

    scene = app.load(ARGS.preset)
    clock = pygame.time.Clock()
    dragging = panning = False
    frames = 0
    shot_at = set()
    if ARGS.shot and ARGS.shotframes:
        shot_at = {int(x) for x in ARGS.shotframes.split(",") if x.strip()}
    t_start = time.time()

    while app.running:
        for ev in pygame.event.get():
            impl.process_event(ev)

            if ev.type == pygame.QUIT:
                app.running = False
                continue
            if ev.type == pygame.VIDEORESIZE:
                renderer.resize(ev.w, ev.h)
                continue

            if ev.type == pygame.KEYDOWN and not io.want_capture_keyboard:
                if ev.key == pygame.K_ESCAPE:
                    app.running = False
                elif ev.key in _SCENE_KEYS:
                    app.load(_SCENE_KEYS[ev.key])
                elif ev.key == pygame.K_r:
                    app.load(app.sim.key)
                elif ev.key == pygame.K_SPACE:
                    app.sim.paused = not app.sim.paused
                elif ev.key == pygame.K_l:
                    app.lensing = not app.lensing
                elif ev.key == pygame.K_f:
                    cam.reset()
                elif ev.key == pygame.K_F11:
                    toggle_fullscreen()
                elif ev.key == pygame.K_g:
                    app.show_gui = not app.show_gui
                elif ev.key == pygame.K_n:
                    app.show_labels = not app.show_labels
                elif ev.key == pygame.K_o:
                    app.show_orbits = not app.show_orbits
                elif ev.key == pygame.K_x:
                    app.spawn_star()
                elif ev.key == pygame.K_v:
                    renderer.gas_on = not renderer.gas_on
                elif ev.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    app.sim.speed = min(app.sim.speed * 1.4, 16.0)
                elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                    app.sim.speed = max(app.sim.speed / 1.4, 0.05)

            if io.want_capture_mouse:
                dragging = panning = False
                continue

            if ev.type == pygame.MOUSEBUTTONDOWN:
                mods = pygame.key.get_mods()
                if ev.button == 1:
                    if mods & pygame.KMOD_SHIFT:
                        panning = True
                    else:
                        dragging = True
                elif ev.button == 2:
                    panning = True
                elif ev.button == 4:
                    cam.dolly(0.88)
                elif ev.button == 5:
                    cam.dolly(1.14)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button in (1, 2):
                    dragging = panning = False
            elif ev.type == pygame.MOUSEMOTION:
                dx, dy = ev.rel
                if panning:
                    cam.pan(dx, dy)
                elif dragging:
                    cam.orbit(-dx * 0.005, dy * 0.005)
            elif ev.type == pygame.MOUSEWHEEL:
                cam.dolly(0.88 ** ev.y if ev.y > 0 else 1.14 ** (-ev.y))

        if not io.want_capture_keyboard:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                cam.dolly(0.97)
            if keys[pygame.K_s]:
                cam.dolly(1.03)
            if keys[pygame.K_a]:
                cam.orbit(0.018, 0.0)
            if keys[pygame.K_d]:
                cam.orbit(-0.018, 0.0)
            if keys[pygame.K_q]:
                cam.orbit(0.0, -0.014)
            if keys[pygame.K_e]:
                cam.orbit(0.0, 0.014)
            if keys[pygame.K_LEFT]:
                cam.pan(9.0, 0.0)
            if keys[pygame.K_RIGHT]:
                cam.pan(-9.0, 0.0)
            if keys[pygame.K_UP]:
                cam.pan(0.0, -9.0)
            if keys[pygame.K_DOWN]:
                cam.pan(0.0, 9.0)

        if not app.sim.paused:
            app.sim.step()
        positions = pos.to_numpy()[:scene.n]
        app.update_tde(positions)
        app.update_gas()

        impl.process_inputs()
        imgui.new_frame()
        scene = app.scene
        if app.show_gui:
            draw_gui(app)
        if app.show_labels or any(app.mission_on.values()):
            draw_labels(app, positions)
        imgui.render()

        # Drawn as gas, the particles that went into the density field must
        # not also be drawn as sprites.  They are a contiguous block at the
        # end, so not drawing them is just a shorter draw call.
        if scene.gas == "volume" and renderer.gas_on:
            vcount = scene.gas_lo
        else:
            vcount = scene.n_base
        _reset_scissor()
        renderer.draw(scene, cam, positions, app.sim.time, app.lensing,
                      vertex_count=vcount, line_kinds=app.visible_lines(),
                      extras=app.extra_ranges())
        impl.render(imgui.get_draw_data())

        pygame.display.flip()
        clock.tick(240)
        app.frame_ms = clock.get_time()
        app.fps = clock.get_fps()

        frames += 1
        if frames in shot_at:
            _save_shot(ctx, renderer.win_w, renderer.win_h, f"{ARGS.shot}_{frames:05d}.png")
        if ARGS.frames and frames >= ARGS.frames:
            app.running = False

    total = time.time() - t_start
    try:
        impl.shutdown()
    except Exception:
        pass          # imgui-bundle teardown trips a stale GL error on exit
    pygame.quit()
    if ARGS.frames:
        print(f"{frames} frames in {total:.2f}s -> {frames / max(total, 1e-6):.1f} fps")


if __name__ == "__main__":
    main()
