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
                              and real space-mission trajectories built as
                              patched conics -- true flyby distances, orbit
                              shapes and escape asymptotes.  The date box in
                              the "Solar System" panel (or --date) places every
                              body where it REALLY is on any day you name,
                              today included, from its published mean
                              longitude; the panel then shows the date ticking
                              forward as the run goes on, because a scene
                              measured in AU and solar masses has a year in it.
                              "Now" (or --date today) goes further and locks
                              the clock to the wall clock, one second per
                              second, so it stays on the present and reads out
                              the real date and time instead of crossing a
                              fortnight of Solar System every second
                              The Oort cloud and the heliosphere are switchable
                              in the same panel: both are far enough out that
                              seeing either means leaving the planets as a knot
                              in the middle
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

import bh_physics as bhp   # Kerr radii, pseudo-Newtonian force, jet power

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--preset", type=int, default=2, choices=(1, 2, 3, 4),
                    help="scene to start in (default: 2, galaxy merger)")
    ap.add_argument("--fullscreen", action="store_true",
                    help="start fullscreen instead of in a resizable window")
    ap.add_argument("--date", default="",
                    help="place the Solar System's planets where they really are "
                         "on this date: YYYY-MM-DD, or 'today'.  Omitted, they are "
                         "scattered to random true anomalies as before")
    ap.add_argument("--real-time", dest="real_time", action="store_true",
                    help="run the clock at wall-clock speed, one second per "
                         "second, instead of racing ahead.  Implied by "
                         "--date today")
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
            gconst: ti.f32, bh_rh: ti.f32, bh_beta: ti.f32):
    """Direct-sum gravity.

    Sources are indices [src_lo, src_hi): the massive bodies.  Every particle
    in [0, n) is accelerated by them, so a scene can mix live self-gravitating
    matter with massless tracers at a fraction of the O(N^2) cost.

    The source set is a *range* rather than a prefix so that the black hole
    scene can give self-gravity to just the one star that is still whole
    while several earlier debris streams are on the board: those are long
    since dominated by the hole, and including them would push the O(N^2)
    inner loop up by the square of the number of stars for no visible gain.

    If bh_rh > 0 the body at index 0 is the hole and is always a source,
    outside the range, through the Artemova-Bjornsson-Novikov (1996)
    pseudo-Newtonian force for a SPINNING hole,
        F = GM / (r^(2 - beta) (r - r_H)^beta),   beta = r_isco / r_H - 1,
    which puts the horizon r_H and the ISCO at their true Kerr radii for the
    hole's spin and makes orbits inside the ISCO plunge.  At zero spin it is
    exactly the Paczynski-Wiita force this used to be, -GM/(r - r_s)^2.
    Callers pass src_lo >= 1 so the hole is not also a Newtonian source.
    """
    for i in range(n):
        a = ti.Vector([0.0, 0.0, 0.0])
        pi = pos[i]
        if bh_rh > 0.0:
            d = pos[0] - pi
            r2 = d.dot(d) + sft2[0]
            r = ti.sqrt(r2)
            rr = ti.max(r - bh_rh, 0.35 * bh_rh)
            a += (gconst * mass[0]
                  / (ti.pow(r, 2.0 - bh_beta) * ti.pow(rr, bh_beta) * r)) * d
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


DISK_TPEAK_SHOWN = 6500.0   # K: white-hot inner disk, see Renderer.disk_tscaled
DISK_TMIN_SHOWN = 3300.0    # K: the outer disk bottoms out at orange, not deep red


def main_sequence_teff(m_sun):
    """Surface temperature of a main-sequence star of m_sun solar masses, K.

    From the mass-luminosity relation (Duric 2004; Salaris & Cassisi 2005)
        L = 0.23 M^2.3        M < 0.43
        L = M^4               0.43 <= M < 2
        L = 1.4 M^3.5         2 <= M < 55
    and the mass-radius relation R = M^0.8 (M <= 1), M^0.57 (M > 1), via
    L = 4 pi R^2 sigma T^4, i.e. T = T_sun (L / R^2)^(1/4).  Gives ~3000 K at
    0.2 M_sun (an M dwarf: red), 5772 K at 1 M_sun (the Sun: yellow-white),
    ~14000 K at 4 M_sun (a B star: blue-white)."""
    m = max(float(m_sun), 0.08)
    if m < 0.43:
        lum = 0.23 * m ** 2.3
    elif m < 2.0:
        lum = m ** 4.0
    else:
        lum = 1.4 * m ** 3.5
    rad = m ** 0.8 if m <= 1.0 else m ** 0.57
    return 5772.0 * (lum / (rad * rad)) ** 0.25


class Kerr:
    """Every radius of a hole of spin a (J c / G M^2), in r_s -- the world
    unit of the black hole scene.  a > 0 turns the same way as the disk
    (which always orbits the way the star is thrown in); a < 0 against it.
    Formulas and checks in bh_physics.py (python bh_physics.py)."""

    def __init__(self, a):
        self.a = bhp.clamp_spin(a)
        self.rh = 0.5 * bhp.horizon(self.a)          # outer event horizon
        self.isco = 0.5 * bhp.isco(self.a)           # innermost stable orbit
        self.rmb = 0.5 * bhp.marginally_bound(self.a)
        self.ph_co = 0.5 * bhp.photon_orbit(self.a)  # photon orbits, with and
        self.ph_counter = 0.5 * bhp.photon_orbit(-self.a)   # against the spin
        self.beta = bhp.isco(self.a) / bhp.horizon(self.a) - 1.0
        self.eta = bhp.efficiency(self.a)
        # Capture: bound gas that gets inside the marginally bound orbit has
        # no turning point left and will cross the horizon; it is taken out
        # there rather than at the horizon, where the force is steep enough
        # for a finite step to fling it back out.  Never inside 1.1 r_H.
        self.r_cap = max(self.rmb, 1.1 * self.rh)

    def omega(self, r):
        """Angular velocity of a circular prograde orbit (BL time), per r_s/c."""
        return 2.0 * bhp.kepler_omega(2.0 * r, self.a)


# --- tidal disruption: spawning a star, accreting it, circularising it ------
#
# The black hole scene starts bare.  A star is spawned as a live,
# self-gravitating Plummer ball parked outside the array's active range until
# then; self-gravity is what lets it hold together on the way in and then lose
# to the tide at pericentre, instead of shearing apart from frame one.

GRAVEYARD = 6.0e4      # where accreted particles are parked: far enough that
                       # the point sprite's distance falloff makes them vanish

# The spawned star is a Plummer ball, and this is its scale radius as a
# fraction of the "star radius" the panel asks for (which is the truncation
# radius, rcut * a).  Named rather than written twice, because the tidal radius
# the panel reports is derived from it: a Plummer sphere has no edge, so the
# radius that belongs in r_t = r (M_bh/M_star)^(1/3) is its half-mass radius,
# 1.305 a.  Those two numbers disagreeing is how the panel came to report a
# tidal radius 19% larger than the star it was actually describing.
STAR_PLUMMER_A = 0.42
STAR_HALF_MASS = 1.305 * STAR_PLUMMER_A

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
              jet_range: ti.f32, r_cap: ti.f32) -> ti.i32:
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
        # r_cap, the marginally bound orbit for this spin (Kerr.r_cap), not
        # the horizon itself: bound gas inside it has no turning point left
        # and is committed to falling in, and capturing it out there keeps it
        # clear of the radius where the force gets steep enough that a finite
        # timestep would slingshot it back out.  Checked every substep for
        # the same reason -- one frame of travel is enough for a fast
        # plunging orbit to dive deep between checks.
        if live and (pos[i] - pos[0]).norm() < r_cap:
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
def k_circularise(lo: ti.i32, hi: ti.i32, dt: ti.f32, circ: ti.f32,
                  alpha: ti.f32, gm: ti.f32, rh: ti.f32, beta: ti.f32,
                  hr0: ti.f32, hr1: ti.f32):
    """Stand-in for the gas physics a collisionless N-body cannot have, at the
    rates physics sets rather than at fixed per-frame fractions.

      * Circularisation.  Real tidal debris only settles into a disk because
        the returning stream shocks against itself and radiates the energy
        away.  The radial (and, lightly, vertical) velocity is damped at a
        fraction `circ` per radian of orbit -- rate circ * Omega(r) -- at
        fixed angular momentum, which drives an eccentric orbit onto the
        circular one with the same L: the disk forms at the circularisation
        radius, about 2 r_p.
      * Viscosity.  An alpha disk (Shakura & Sunyaev 1973): nu = alpha (H/R)^2
        r^2 Omega moves gas inward at v_r = -3 nu / 2r, i.e. a circular orbit
        loses angular momentum at the fractional rate (3/4) alpha (H/R)^2
        Omega.  H/R = hr0 + hr1 / r is the thickness the disk is drawn with.
        This is what makes the disk drain into the hole, on its viscous time
        -- not a fixed fraction per frame."""
    for i in range(lo, hi):
        d = pos[i] - pos[0]
        r = d.norm()
        # |y| < 0.3 r keeps this to material that is actually in the disk --
        # jet particles climbing out along the poles must not be dragged back
        # down into the plane by it
        if r > 1.05 * rh and r < 0.5 * GRAVEYARD and abs(d[1]) < 0.3 * r:
            om = ti.sqrt(gm * ti.pow(r, beta - 1.0)
                         / ti.pow(ti.max(r - rh, 1e-4), beta)) / r
            dv = vel[i] - vel[0]
            rhat = d / r
            fc = 1.0 - ti.exp(-circ * om * dt)
            dv -= rhat * (dv.dot(rhat) * fc)
            # only lightly: the vertical motion is what gives the disk its
            # thickness, and damping it as hard as the radial component
            # flattens the disk onto the midplane exactly
            dv[1] -= dv[1] * (fc * 0.10)
            hr = hr0 + hr1 / r
            ft = 1.0 - ti.exp(-0.75 * alpha * hr * hr * om * dt)
            v_t = dv - rhat * dv.dot(rhat)
            dv -= v_t * ft
            vel[i] = vel[0] + dv


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
gas_u8 = ti.Vector.field(2, ti.u8, shape=(GAS_NZ, GAS_NY, GAS_NX))
GAS_DREF = 420.0         # density that saturates the byte: a little above
                         # what the core of an intact star reaches
# Channel 1 of the grid: the fraction of the gas in each cell that has joined
# the disk.  A star and the stream it is drawn into keep the star's colour
# wherever they are; only gas that has actually settled onto a disk orbit
# takes the disk's.  Kept per particle and latched: gas that has been shocked
# into the disk stays disk gas.
in_disk = ti.field(ti.f32, shape=MAX_N)
gas_s = ti.field(ti.f32, shape=(GAS_NZ, GAS_NY, GAS_NX))
gas_t = ti.field(ti.f32, shape=(GAS_NZ, GAS_NY, GAS_NX))


@ti.kernel
def k_mark_disk(lo: ti.i32, hi: ti.i32, gm: ti.f32, rh: ti.f32, beta: ti.f32,
                core: ti.types.vector(3, ti.f32), core_r2: ti.f32):
    """Latch in_disk = 1 for particles on near-circular orbits -- the same test
    App.update_tde builds the disk from -- except inside a star still whole."""
    for i in range(lo, hi):
        if in_disk[i] < 0.5:
            d = pos[i] - pos[0]
            r = d.norm()
            if r > 1.05 * rh and r < 0.5 * GRAVEYARD and (pos[i] - core).norm_sqr() > core_r2:
                dv = vel[i] - vel[0]
                vc = ti.sqrt(gm * ti.pow(r, beta - 1.0) / ti.pow(r - rh, beta))
                vr = ti.abs(dv.dot(d)) / r
                kap = d.cross(dv).norm() / (r * vc)
                if vr < 0.30 * vc and kap > 0.75 and kap < 1.3:
                    in_disk[i] = 1.0


@ti.kernel
def k_fill_in_disk(lo: ti.i32, hi: ti.i32, v: ti.f32):
    for i in range(lo, hi):
        in_disk[i] = v


@ti.kernel
def k_gas_quantise(src: ti.template(), srcs: ti.template()):
    for I in ti.grouped(src):
        v = ti.sqrt(ti.min(src[I] * (1.0 / GAS_DREF), 1.0))
        f = ti.min(srcs[I] / ti.max(src[I], 1e-9), 1.0)
        gas_u8[I] = ti.Vector([ti.cast(v * 255.0 + 0.5, ti.u8),
                               ti.cast(f * 255.0 + 0.5, ti.u8)])


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
        gas_s[I] = 0.0
    for i in range(lo, hi):
        q = pos[i]
        sh = in_disk[i]
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
                gas_s[iz + dz, iy + dy, ix + dx] += amp * wx * wy * wz * sh


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
    k_gas_blur(gas_s, gas_t, 0)
    k_gas_blur(gas_t, gas_s, 1)
    k_gas_blur(gas_s, gas_t, 2)
    k_gas_quantise(gas_b, gas_t)
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
                 gas="", gas_half=(GAS_RX, GAS_RY), gas_lo=1, extras=(),
                 epoch_jd=None):
        self.name = name
        # Julian day the scene was built for, or None if its bodies were placed
        # at arbitrary phases.  Set, it makes sim time a real elapsed time and
        # lets the panel put a calendar date on the frame (see YEAR_UNITS).
        self.epoch_jd = float(epoch_jd) if epoch_jd is not None else None
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


def ecl_to_sim(v):
    """Ecliptic (X, Y, Z-toward-north) -> the scene's (x, y-up, z).  Every
    piece of real orbital geometry in this file comes in ecliptic coordinates
    and goes through here, so the convention is written down once instead of
    being re-derived, differently, at each site that needs it.

    This alone is a mirror image; the reflection is undone for the whole scene
    at once at the end of preset_solar_system.  See the note there."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], v[2], v[1]])


def orbit_plane_axes(i_deg, node_deg):
    """The two unit vectors that span an orbit plane, in sim axes: the first
    points at the ascending node, the second 90 degrees along the direction of
    motion.  A body at argument of latitude u then sits at

        cos(u) * node_axis + sin(u) * ahead_axis

    and u simply advancing is a body going round the right way -- including
    backwards, with no flag to say so, because an inclination past 90 degrees
    turns the second axis around on its own.  That is how Triton and Charon
    get their real retrograde orbits here: from their real inclinations."""
    i, om = math.radians(i_deg), math.radians(node_deg)
    node = np.array([math.cos(om), math.sin(om), 0.0])
    ahead = np.array([-math.cos(i) * math.sin(om),
                      math.cos(i) * math.cos(om), math.sin(i)])
    return ecl_to_sim(node), ecl_to_sim(ahead)


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
    return ecl_to_sim([Xa, Ya, Za]), ecl_to_sim([Vxa, Vya, Vza])


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
    return np.stack([Xa, Za, Ya], axis=1).astype(np.float32)   # ecl_to_sim, vectorised


# ---------------------------------------------------------------------------
# 1) Solar System disk
# ---------------------------------------------------------------------------

AU_UNITS = 10.0           # scene units per astronomical unit (see preset_solar_system)
R_SUN_AU = 0.00465047     # one solar radius in AU -- Parker's perihelia are published in these


# --- putting the planets where they really are on a given date --------------
#
# Every body in the table below carries, besides its orbit's shape and
# orientation, its MEAN LONGITUDE: the angle it would be at if it swept the
# orbit at a constant rate, published as a value at J2000 plus a rate per
# century.  That one extra number per body is the whole difference between a
# solar system whose planets are scattered at random and one that shows where
# they actually were -- or will be -- on a particular day:
#
#     date -> Julian day -> centuries since J2000 -> mean longitude L
#          -> mean anomaly M = L - peri -> Kepler's equation -> true anomaly
#
# and a true anomaly is exactly what kepler_state already takes, so a dated
# scene is the same scene with one number per planet solved for instead of
# drawn from the RNG.  Nothing about the physics changes: the bodies are still
# handed to the same integrator and still move under their mutual gravity from
# that starting configuration onward.
#
# Only the mean longitude is advanced with the date.  The other five elements
# drift too, but at rates (Saturn's perihelion, the fastest, moves 0.42 deg a
# century) that are invisible here against a Saturn drawn 1.5 units wide, while
# the mean longitude runs through thousands of degrees over the same span and
# is the entire reason one date looks different from another.

J2000_JD = 2451545.0        # 2000 Jan 1.5 TT -- the epoch the elements are referred to
DAYS_PER_CENTURY = 36525.0   # a Julian century, the unit the element rates use

# Days in one revolution of the scene's own Earth.  With G = 1, M_sun = 1 and
# a = AU_UNITS, Kepler's third law makes that the SIDEREAL year, 365.2564 d --
# not the Julian year of 365.25 that the element rates above are quoted in.
# The two differ by 1.7e-5, which sounds like nothing and is not: it is a
# systematic rate error in the clock rather than a wobble, so it accumulates
# to two thirds of a day per century of run time, and the clock's whole
# purpose is to be the date rather than approximately the date.
SIM_YEAR_DAYS = 365.256363

# One Julian year in sim time units.  This is not a fitted constant: the
# preset runs with G = 1, M_sun = 1 and Earth at a = AU_UNITS, so Kepler's
# third law fixes a year as one revolution at that radius and leaves nothing to
# choose.  It is what lets a dated run show a real date ticking forward rather
# than a bare step count.
YEAR_UNITS = 2.0 * math.pi * AU_UNITS ** 1.5


def julian_day(year, month, day):
    """Proleptic-Gregorian calendar date -> Julian day at 00:00 UT, by the
    usual integer-arithmetic form.  day may carry a fraction."""
    y, m = int(year), int(month)
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + float(day) + b - 1524.5)


def calendar_date(jd):
    """Julian day -> (year, month, day), the exact inverse of julian_day and
    proleptic Gregorian for the same reason: a run started in 1500 should read
    back the date it was given, not the Julian-calendar date that fell on the
    same day."""
    z = math.floor(jd + 0.5)
    f = (jd + 0.5) - z
    alpha = math.floor((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - math.floor(alpha / 4.0)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = int(e - 1 if e < 14 else e - 13)
    year = int(c - 4716 if month > 2 else c - 4715)
    return year, month, int(math.floor(day))


def clock_time(jd):
    """Time of day at a Julian day, as (hour, minute, second) UTC.  A Julian
    day starts at noon, which is what the half-day offset is undoing."""
    secs = int(round(((jd + 0.5) % 1.0) * 86400.0)) % 86400
    return secs // 3600, (secs // 60) % 60, secs % 60


def today_jd():
    """This instant, as a Julian day -- time of day included, so a scene built
    from it starts at the actual moment and not at midnight."""
    tm = time.gmtime()
    frac = (tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec) / 86400.0
    return julian_day(tm.tm_year, tm.tm_mon, tm.tm_mday + frac)


def parse_date(text):
    """'YYYY-MM-DD', or 'today'/'now', -> Julian day.  Raises ValueError on
    anything else, so a mistyped --date is a message rather than a scene
    quietly built for the wrong day."""
    s = str(text).strip().lower()
    if s in ("today", "now"):
        return today_jd()
    parts = s.replace("/", "-").split("-")
    if len(parts) != 3:
        raise ValueError(f"cannot read {text!r} as a date (want YYYY-MM-DD or 'today')")
    y, m, d = (int(p) for p in parts)
    if not 1 <= m <= 12 or not 1 <= d <= 31:
        raise ValueError(f"{text!r} is not a real date")
    return julian_day(y, m, d)


# --date, resolved once here rather than when the scene is built, so a
# mistyped one is a line of text before anything starts instead of a traceback
# out of a window that has already opened.
try:
    ARGS_EPOCH_JD = parse_date(ARGS.date) if ARGS.date else None
except ValueError as _exc:
    raise SystemExit(f"--date: {_exc}")

# Asking for today is asking to watch the Solar System as it is, so the clock
# comes up locked to the wall clock rather than sprinting; a specific date has
# no such implication and runs at the usual speed unless --real-time says not to.
ARGS_EPOCH_IS_NOW = ARGS.date.strip().lower() in ("today", "now")
ARGS_REAL_TIME = bool(ARGS.real_time) or ARGS_EPOCH_IS_NOW

# The most wall-clock time one frame will ever integrate.  Frames stop arriving
# while a window is minimised or a laptop is asleep, and without a ceiling the
# first frame back would try to swallow the whole absence in a single step --
# a step big enough to be wrong, and on a long enough sleep to fling the
# planets off.  Past this the missed time is simply dropped: the clock falls
# behind, visibly, and the Now button puts it back.
REALTIME_MAX_CATCHUP = 3600.0


def solve_kepler(m_deg, e):
    """Mean anomaly (degrees) -> eccentric anomaly (radians), by Newton's
    method on Kepler's equation M = E - e sin E.  The transcendental step no
    closed form gets past, and the one place a dated scene costs anything at
    all -- a dozen iterations, once per planet, at load."""
    m = math.radians((m_deg + 180.0) % 360.0 - 180.0)
    ea = m + e * math.sin(m)
    for _ in range(64):
        step = (ea - e * math.sin(ea) - m) / (1.0 - e * math.cos(ea))
        ea -= step
        if abs(step) < 1e-13:
            break
    return ea


# Saturn's equatorial plane in ecliptic terms, from the IAU pole (right
# ascension 40.589 deg, declination 83.537 deg) rotated out of the equatorial
# frame: the pole tips 28.05 deg from the ecliptic pole and the plane crosses
# the ecliptic at longitude 169.53.  The rings and every regular moon of
# Saturn lie in it, and Horizons agrees -- Enceladus, Rhea and Titan come back
# with inclinations of 28.05, 28.24 and 27.72 about nodes of 169.5, 169.0 and
# 169.2, which is those moons' own degree of tilt out of exactly this plane.
SATURN_EQUATOR_INCL = 28.049
SATURN_EQUATOR_NODE = 169.53

# What this model of the moons is worth: the angle between the direction each
# moon is placed in and the direction JPL's ephemeris puts it in, worst case
# over three dates spread across 1997-2044.
#
#   Charon    0.02      Enceladus 1.33      Triton  3.25
#   Titania   0.11      Europa    1.70      Phobos  4.14
#   Ganymede  0.29      Deimos    2.94      Titan   5.20
#   Io        0.46      Rhea      0.43      Moon    9.59
#   Callisto  0.85
#
# Under two degrees is most of them, and two degrees is far less than the
# width of the planet they are drawn beside.  The three that are worse are
# worse for reasons a circle cannot fix: Titan's orbit is eccentric enough
# (0.029) that a circle is off by twice that in angle on its own; Triton's
# steeply inclined plane precesses; and the Moon carries perturbations the Sun
# puts into it -- evection at 1.27 degrees, the variation at 0.66 -- that no
# fixed circle has anywhere to put.  Ten degrees of a 27-day orbit is about
# three quarters of a day: the Moon is on the right side of the Earth, not at
# the right hour.


def true_anomaly_at(jd, e, peri_deg, l0_deg, dl_cy):
    """True anomaly (radians) on Julian day jd of a body whose mean longitude
    is l0_deg at J2000 and advances dl_cy degrees per Julian century."""
    t = (jd - J2000_JD) / DAYS_PER_CENTURY
    ea = solve_kepler(l0_deg + dl_cy * t - peri_deg, e)
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * ea),
                            math.sqrt(1.0 - e) * math.cos(0.5 * ea))


# name, semi-major axis (AU), eccentricity, inclination, longitude of the
# ascending node, LONGITUDE OF PERIHELION (all degrees), mean longitude at
# J2000 and its rate in degrees per Julian century, mass (solar masses),
# draw radius, colour, and moons as (name, orbit radius in multiples of the
# planet's own drawn radius, drawn size, colour, sidereal period in days,
# inclination to the ECLIPTIC, longitude of the ascending node, and
# argument of latitude at J2000 -- the angle round the orbit from that
# node, which is what puts a moon on the correct side of its planet).
#
# Only the orbit radius is a drawn quantity; the other four are measured.
# Inclination and node are the osculating ecliptic values at J2000 from
# JPL Horizons, which for a regular satellite is its planet's equator --
# hence Uranus's moon at 97.8 degrees and Neptune's at 130.3, tipped and
# retrograde exactly as far as the real ones are.  Period and argument of
# latitude were then fitted to Horizons positions sampled from 1960 to
# 2050, rather than taken from published period tables, because the angle
# here is measured from a FIXED node while a published sidereal period is
# not always: over ninety years the difference is whole revolutions.  The
# fit residuals are what this model is worth, and they are listed against
# each moon beside SATURN_EQUATOR_INCL above.
#
# The eight planets and Pluto are Standish's table for the approximate
# positions of the major planets, fitted over 1800-2050 and good to well
# under a thousandth of an AU for the inner planets and a few hundredths
# for the outer ones across that span -- far finer than a planet here is
# drawn.  The four remaining dwarf planets are their current osculating
# elements from JPL's small-body database, converted to the same form.
#
# Note "longitude of perihelion", not "argument of perihelion": the two
# differ by the node, and the argument kepler_state wants is recovered as
# peri - node below.  This column USED to be fed straight in as if it were
# the argument, which rotated each planet's ellipse within its own plane by
# its node -- harmless while the bodies sat at random anomalies, and wrong
# the moment a real date is asked for.
SOLAR_BODIES = [
    ("Mercury", 0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628,
     252.25032350, 149472.67411175,
     1.66e-7, 0.45, (1.00, 0.86, 0.66), []),
    ("Venus", 0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718,
     181.97909950, 58517.81538729,
     2.45e-6, 0.70, (1.00, 0.82, 0.50), []),
    # Earth's row is really the Earth-Moon barycentre's, which is what the
    # published table tracks and what the Sun actually pulls on.
    ("Earth", 1.00000261, 0.01671123, -0.00001531, 0.0, 102.93768193,
     100.46457166, 35999.37244981,
     3.00e-6, 0.72, (0.45, 0.68, 1.00), [
        ("Moon", 1.9, 0.11, (0.80, 0.80, 0.78),
         27.3221263, 5.2403, 123.9581, 95.339),
    ]),
    ("Mars", 1.52371034, 0.09339410, 1.84969142, 49.55953891, -23.94362959,
     -4.55343205, 19140.30268499,
     3.23e-7, 0.58, (1.00, 0.48, 0.32), [
        ("Phobos", 1.5, 0.045, (0.65, 0.55, 0.48),
         0.3189101, 26.0567, 84.8151, 172.502),
        ("Deimos", 2.1, 0.040, (0.60, 0.58, 0.55),
         1.2624408, 27.5694, 83.6693, 217.091),
    ]),
    ("Ceres", 2.76555260, 0.07969230, 10.58802780, 80.24862682, 153.54284135,
     158.74556430, 7827.47006000,
     4.72e-10, 0.14, (0.62, 0.60, 0.58), []),
    ("Jupiter", 5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983,
     34.39644051, 3034.74612775,
     9.55e-4, 1.70, (0.95, 0.80, 0.62), [
        ("Io", 2.3, 0.15, (0.90, 0.78, 0.45),
         1.7691377, 2.2126, 336.8524, 41.167),
        ("Europa", 3.0, 0.13, (0.85, 0.80, 0.72),
         3.5511813, 1.7910, 332.6287, 239.648),
        ("Ganymede", 3.9, 0.17, (0.72, 0.66, 0.58),
         7.1545539, 2.2141, 343.1728, 236.862),
        ("Callisto", 5.0, 0.16, (0.48, 0.44, 0.42),
         16.6890147, 2.0169, 337.9426, 100.948),
    ]),
    ("Saturn", 9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831,
     49.95424423, 1222.49362201,
     2.86e-4, 1.50, (0.98, 0.88, 0.66), [
        ("Enceladus", 2.6, 0.075, (0.92, 0.94, 0.96),
         1.3702183, 28.0520, 169.5066, 142.301),
        ("Rhea", 3.4, 0.095, (0.80, 0.78, 0.72),
         4.5175025, 28.2414, 168.9842, 12.676),
        ("Titan", 4.6, 0.19, (0.90, 0.72, 0.42),
         15.9455490, 27.7183, 169.2392, 327.251),
    ]),
    ("Uranus", 19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630,
     313.23810451, 428.48202785,
     4.37e-5, 1.05, (0.62, 0.92, 0.96), [
        ("Titania", 2.6, 0.09, (0.72, 0.78, 0.85),
         8.7058689, 97.8184, 167.6178, 276.500),
    ]),
    ("Neptune", 30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227,
     -55.12002969, 218.45945325,
     5.15e-5, 1.02, (0.42, 0.60, 1.00), [
        ("Triton", 2.8, 0.10, (0.62, 0.72, 0.95),
         5.8768440, 130.2614, 215.8591, 74.690),
    ]),
    ("Pluto", 39.48211675, 0.24882730, 17.14001206, 110.30393684, 224.06891629,
     238.92903833, 145.20780515,
     6.55e-9, 0.26, (0.80, 0.68, 0.58), [
        ("Charon", 2.2, 0.13, (0.72, 0.70, 0.68),
         6.3872221, 112.8908, 227.3917, 321.251),
    ]),
    ("Haumea", 43.06029000, 0.19444300, 28.20847400, 121.78605600, 2.47660300,
     192.00769000, 127.40277000,
     2.02e-9, 0.19, (0.88, 0.90, 0.92), []),
    ("Makemake", 45.57093300, 0.15888900, 29.02785600, 79.29483400, 16.38710700,
     155.39033000, 117.02063000,
     1.56e-9, 0.18, (0.72, 0.48, 0.38), []),
    ("Eris", 67.93394700, 0.43823900, 43.92582800, 36.00477000, 186.79969400,
     21.57806000, 64.29305000,
     8.35e-9, 0.25, (0.85, 0.85, 0.88), []),
]


def body_longitude(name, jd):
    """Heliocentric ecliptic longitude of a named body on a Julian day, in
    degrees.  Pulled straight off the same table and the same solver that
    place the body in the scene, so a mission path aimed with this cannot
    drift away from the planet it is aimed at."""
    for row in SOLAR_BODIES:
        if row[0] != name:
            continue
        _, a_au, e, i_deg, node_deg, peri_deg, l0_deg, dl_cy = row[:8]
        nu = true_anomaly_at(jd, e, peri_deg, l0_deg, dl_cy)
        p = kepler_state(1.0, e, i_deg, node_deg, peri_deg - node_deg, nu)[0]
        # kepler_state hands back scene axes, where before the preset's final
        # mirror the azimuth in the x-z plane IS the ecliptic longitude
        return math.degrees(math.atan2(p[2], p[0])) % 360.0
    raise KeyError(name)


def _rodrigues(v, axis, ang):
    """Rotate vector v about a unit axis by ang radians."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v) * (1.0 - c))


def _nu_at_r(q, e, r):
    """True anomaly in 0..pi at which a conic of perihelion distance q and
    eccentricity e sits at radius r -- the inverse of r = q(1+e)/(1+e cos nu).
    Clamped, so asking an ellipse for a radius past its aphelion gives the
    aphelion (nu = pi) rather than a domain error."""
    if e < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, (q * (1.0 + e) / r - 1.0) / e)))


class Cruise:
    """A heliocentric trajectory assembled leg by leg out of TRUE conic arcs.

    Between encounters a spacecraft is a two-body problem and nothing else: it
    coasts along a conic section about the Sun, an ellipse if it is bound and a
    hyperbola if it is not.  A gravity assist then changes its velocity over a
    few days while leaving it essentially where it was, so the next conic
    starts exactly where the previous one ended, with a different shape and, in
    general, an orbit plane tilted about the Sun-to-spacecraft line.  Chaining
    conics that way is how these trajectories are really designed, and building
    the drawn paths the same way is what makes them come out the right SHAPE --
    a smooth curve threaded through the published flyby distances cannot,
    because it has no way to know that a probe whips through perihelion and
    crawls through aphelion, that a bound orbit closes on itself instead of
    drifting sideways, or that an escaping one straightens onto a fixed
    asymptote instead of curving forever.

    The chain carries a current position and a current orbit normal.  Each leg
    optionally tilts the plane about the current radius vector -- the flyby
    point lies on that axis, so it does not move, which is exactly the property
    that lets the legs join with no seam and no fudging -- then sweeps a conic
    through a span of true anomaly, anchored so the leg begins on the point the
    last one ended at.

    Heliocentric DISTANCES, orbit shapes, plane tilts and the order of
    encounters are the real published ones, and aim_at then turns the finished
    chain until its flybys sit on the planets they really were flybys of, on
    the dates they really happened.  So the bend in Voyager 1's path is not
    near Saturn's orbit, it is on Saturn -- where Saturn stood in November
    1980.  Set the date to that November and the planet is waiting at the bend.

    Which also means the paths do NOT follow the planets around.  A trajectory
    is fixed in space, drawn where it was flown; the planets move, and meet it
    only on the dates they met it.  Every other date shows them apart, which is
    the truth about a spacecraft that went past forty years ago.
    """

    def __init__(self, r0_au, az_deg):
        th = math.radians(az_deg)
        self.p = np.array([math.cos(th), 0.0, math.sin(th)], dtype=np.float64) * r0_au
        # scene axes are (x, up, z); this is the normal that makes the motion
        # prograde -- the same sense the planets are placed going around in
        self.h = np.array([0.0, -1.0, 0.0])
        self.pts = [self.p.copy()]
        # Index in pts of every joint between legs -- which is to say of every
        # encounter, since a leg is exactly the coast between two of them.
        # These are the points that have to land on a planet once the path is
        # turned to face the right way; see align_to_encounters.
        self.joints = [0]

    def leg(self, q, e, nu_end_deg, nu_start_deg=None, tilt_deg=0.0, n=180):
        """Sweep one conic arc: q perihelion distance in AU, e eccentricity,
        the true anomalies in degrees, tilt_deg the gravity assist's rotation
        of the orbit plane about the Sun-to-spacecraft line at the joint."""
        r0 = float(np.linalg.norm(self.p))
        if abs(tilt_deg) > 1e-9:
            axis = self.p / r0
            self.h = _rodrigues(self.h, axis, math.radians(tilt_deg))
            self.h /= np.linalg.norm(self.h)
        if nu_start_deg is None:
            nu_start_deg = math.degrees(_nu_at_r(q, e, r0))
        nu0, nu1 = math.radians(nu_start_deg), math.radians(nu_end_deg)
        # perihelion direction: the current point wound back by its own true
        # anomaly.  peri stays perpendicular to h because p is, so the full
        # Rodrigues formula collapses to the two terms used below.
        peri = _rodrigues(self.p / r0, self.h, -nu0)
        side = np.cross(self.h, peri)
        nu = np.linspace(nu0, nu1, max(int(n), 2))
        r = q * (1.0 + e) / (1.0 + e * np.cos(nu))
        pts = (peri[None, :] * np.cos(nu)[:, None]
               + side[None, :] * np.sin(nu)[:, None]) * r[:, None]
        self.pts.extend(pts[1:])
        self.p = pts[-1].copy()
        self.joints.append(len(self.pts) - 1)
        return self

    def coast_to(self, q, e, r_end_au, tilt_deg=0.0, n=180):
        """The common case: coast outward along a conic from wherever the chain
        is to the radius of the next encounter.  A leg that instead runs in
        past an apsis has to name its own true anomalies, since the radius
        alone no longer says where on the conic either end of it sits."""
        r0 = float(np.linalg.norm(self.p))
        return self.leg(q, e, math.degrees(_nu_at_r(q, e, r_end_au)),
                        nu_start_deg=math.degrees(_nu_at_r(q, e, r0)),
                        tilt_deg=tilt_deg, n=n)

    def revolution(self, q, e, tilt_deg=0.0, n=260):
        """One complete turn of a closed orbit, starting and ending on the
        point the chain is at -- a bound spacecraft comes back to where it was,
        which is the whole difference between an orbit and a trajectory."""
        r0 = float(np.linalg.norm(self.p))
        nu0 = math.degrees(_nu_at_r(q, e, r0))
        return self.leg(q, e, nu0 + 360.0, nu_start_deg=nu0, tilt_deg=tilt_deg, n=n)

    def aim_at(self, encounters):
        """Turn the whole trajectory about the ecliptic pole so its flybys land
        on the planets they really were flybys OF.

        Everything up to here fixes a trajectory's SHAPE -- its distances, its
        conic eccentricities, the angle each leg sweeps -- but leaves it facing
        an arbitrary direction, because a conic built from distances has no
        idea which way round the Sun it should be pointing.  Given the real
        date of each encounter, the planet's real longitude on that date is a
        lookup, and one rotation is then the only freedom left.

        It is only one rotation for the whole path, and that is the honest
        limit: where each flyby sits RELATIVE to the others is already decided
        by the shape, so a mission whose legs sweep slightly the wrong angle
        cannot have every flyby land at once, and the fit shares the error out
        instead of hiding it in one place.  The share is weighted by how far
        out each encounter is, since a degree of error at Saturn is nine times
        the miss that a degree at Earth is, and the miss is what shows.

        The launch point is deliberately not fitted.  It is where the first leg
        happens to start rather than something the conics were built to
        reproduce, and letting it vote drags the flybys -- the points a viewer
        can actually check against a planet -- off by more than it gains.
        """
        pts = np.asarray(self.pts, dtype=np.float64)
        sin_sum = cos_sum = 0.0
        for joint, body, jd in encounters:
            p = pts[self.joints[joint]]
            r = math.hypot(p[0], p[2])
            delta = math.radians(body_longitude(body, jd)
                                 - math.degrees(math.atan2(p[2], p[0])))
            sin_sum += r * math.sin(delta)
            cos_sum += r * math.cos(delta)
        ang = math.atan2(sin_sum, cos_sum)
        ca, sa = math.cos(ang), math.sin(ang)
        x, z = pts[:, 0].copy(), pts[:, 2].copy()
        pts[:, 0] = x * ca - z * sa
        pts[:, 2] = x * sa + z * ca
        self.pts = [p for p in pts]
        return self

    def path(self):
        return (np.asarray(self.pts, dtype=np.float64) * AU_UNITS).astype(np.float32)


# Parker Solar Probe's seven Venus gravity assists, each trading orbital energy
# for a lower perihelion: (perihelion in SOLAR RADII, aphelion in AU).  The
# perihelia are the published step-down -- 35.7 solar radii on the first
# encounter in Nov 2018, 9.86 from 24 Dec 2024.  That last one is measured
# from the Sun's CENTRE, which is what an orbit is: it puts Parker 6.1 million
# km above the surface, the figure the mission quotes, and closer to a star
# than anything else ever built.  The aphelia follow from the
# orbital periods over the same steps (150 days at the start, 88 at the end);
# they walk in from just short of Earth's orbit onto Venus's, which is the
# point a Venus assist can no longer improve on.
PSP_ORBITS = [(35.7, 0.937), (27.9, 0.877), (20.3, 0.818), (16.0, 0.746),
              (13.3, 0.736), (11.4, 0.730), (9.86, 0.7233)]


def _mission_paths():
    """Real mission trajectories, built as patched conics and then aimed at the
    planets they really flew past -- see Cruise and Cruise.aim_at.

    How close each drawn flyby comes to its planet on the encounter date,
    measured in the finished scene:

        Parker/Venus 0.04   Voyager 2/Jupiter 0.08   Voyager 1/Jupiter 0.10
        Voyager 1/Saturn 0.11   Pioneer 10/Jupiter 0.11   NH/Jupiter 0.11
        Juno/Jupiter 0.21   Pioneer 11/Saturn 0.29   Cassini/Saturn 1.48
        Voyager 2/Neptune 2.88                                        (AU)

    A tenth of an AU at Saturn is a hundredth of the radius of its orbit: the
    line goes through the planet.  The two that miss are the two whose drawn
    shape cannot be turned into agreement, because one rotation cannot fix a
    leg that sweeps the wrong angle -- Voyager 2 accumulates a few degrees over
    four flybys, and Cassini's inner-system loops are the most simplified part
    of any path here.
    """
    out = []

    def add(name, color, c):
        out.append({"name": name, "kind": "mission", "color": color, "pos": c.path()})

    # --- Voyager 1: Jupiter Mar 1979, Saturn Nov 1980, where the Titan flyby
    # throws it up and out of the ecliptic.  Its heliocentric orbit after
    # Saturn is a hyperbola of eccentricity 3.71 whose perihelion IS the Saturn
    # encounter, inclined 35.8 deg, which puts the outbound asymptote at
    # ecliptic latitude +35 -- where Voyager 1 is really heading.  Heliopause
    # crossing at 121.6 AU in Aug 2012; drawn out to the 170 AU it has reached,
    # still the most distant object we have made.
    c = Cruise(1.0, 0.0)
    c.coast_to(1.00, 0.815, 5.2)
    c.coast_to(4.90, 1.350, 9.5, tilt_deg=-2.0)
    c.coast_to(9.50, 3.715, 170.0, tilt_deg=-34.8, n=260)
    c.aim_at([(1, "Jupiter", julian_day(1979, 3, 5)),
              (2, "Saturn", julian_day(1980, 11, 12))])
    add("Voyager 1", (0.16, 0.46, 0.52), c)

    # --- Voyager 2: the Grand Tour -- Jupiter Jul 1979, Saturn Aug 1981,
    # Uranus Jan 1986, Neptune Aug 1989, still the only visit either ice giant
    # has had.  It stays near the ecliptic the whole way across, because that
    # is where all four planets are; the plunge south comes only at the end,
    # where the Neptune flyby was aimed over the planet's north pole to reach
    # Triton and left Voyager 2 on an orbit inclined 78.8 deg with its
    # asymptote at ecliptic latitude -48.  Heliopause at 119 AU in Nov 2018;
    # drawn out to 142 AU.
    c = Cruise(1.0, 42.0)
    c.coast_to(1.00, 0.810, 5.2)
    c.coast_to(4.95, 1.300, 9.5, tilt_deg=1.0)
    c.coast_to(9.30, 1.620, 19.2, tilt_deg=2.5)
    c.coast_to(15.0, 1.300, 30.1, tilt_deg=1.0)
    c.coast_to(20.45, 6.285, 142.0, tilt_deg=78.8, n=260)
    c.aim_at([(1, "Jupiter", julian_day(1979, 7, 9)),
              (2, "Saturn", julian_day(1981, 8, 26)),
              (3, "Uranus", julian_day(1986, 1, 24)),
              (4, "Neptune", julian_day(1989, 8, 25))])
    add("Voyager 2", (0.48, 0.20, 0.48), c)

    # --- Pioneer 10: first through the asteroid belt, first past Jupiter
    # (Dec 1973), first onto an escape trajectory out of the Solar System.
    # Jupiter left it on a hyperbola of eccentricity 1.73 barely 3 deg out of
    # the ecliptic, aimed at Aldebaran.  Drawn only as far as the 80 AU it had
    # reached when the last contact came in on 23 Jan 2003, which is the last
    # place anyone actually knows it to have been: it has certainly coasted on
    # since, but only the two Voyagers have a heliopause crossing anybody
    # measured, and extending a silent spacecraft across that boundary would
    # be drawing a guess in the same line weight as a fact.
    c = Cruise(1.0, -150.0)
    c.coast_to(0.99, 0.790, 5.2)
    c.coast_to(5.00, 1.733, 80.0, tilt_deg=-3.1, n=260)
    c.aim_at([(1, "Jupiter", julian_day(1973, 12, 4))])
    add("Pioneer 10", (0.24, 0.48, 0.24), c)

    # --- Pioneer 11: the one that took the long way round.  Jupiter (Dec 1974)
    # threw it back up over the Solar System on an ellipse whose perihelion is
    # the Jupiter encounter and whose aphelion is Saturn -- half a revolution
    # and nearly five years later, on the far side of the Sun.  So it climbs
    # to 16 deg above the ecliptic mid-crossing and comes back down to meet
    # Saturn in Sep 1979, which then sent it out at ecliptic latitude +12.6.
    # Drawn, like Pioneer 10, only to where it was last heard from -- 44.7 AU,
    # out among the Kuiper belt, in Sep 1995.
    c = Cruise(1.0, -160.0)
    c.coast_to(0.99, 0.800, 5.2)
    c.leg(5.20, 0.2925, 180.0, nu_start_deg=0.0, tilt_deg=-15.6, n=240)
    c.coast_to(9.40, 2.147, 44.7, tilt_deg=-28.4, n=240)
    c.aim_at([(1, "Jupiter", julian_day(1974, 12, 3)),
              (2, "Saturn", julian_day(1979, 9, 1))])
    add("Pioneer 11", (0.40, 0.46, 0.17), c)

    # --- New Horizons: Jupiter Feb 2007, Pluto Jul 2015 at 32.9 AU, Arrokoth
    # Jan 2019 at 43.4 AU.  Neither Kuiper belt flyby bent it measurably -- the
    # bodies are far too small -- so everything past Jupiter is one unbroken
    # hyperbola of eccentricity 1.41 and the two encounters are simply points
    # it sails through.  Still transmitting, and drawn out to the 66 AU it has
    # reached; the heliopause is another fifty AU ahead of it.
    c = Cruise(1.0, 15.0)
    c.coast_to(0.99, 0.980, 5.2)
    c.coast_to(2.20, 1.410, 66.0, tilt_deg=2.3, n=260)
    c.aim_at([(1, "Jupiter", julian_day(2007, 2, 28))])
    add("New Horizons", (0.52, 0.30, 0.10), c)

    # --- Cassini-Huygens: the VVEJGA tour, Venus-Venus-Earth-Jupiter Gravity
    # Assist.  It could not reach Saturn directly, so it went inward first --
    # Venus Apr 1998, then a long loop out to 1.58 AU and back for a second
    # Venus pass Jun 1999, Earth two months after that, Jupiter Dec 2000,
    # Saturn orbit insertion Jul 2004.  The 13-year, 294-orbit tour of Saturn
    # that follows is not drawn: its widest apoapsis is well under a hundredth
    # of the Sun-Saturn distance, so at this scale the entire tour is one point
    # -- the point this path ends on.
    c = Cruise(1.0, 170.0)
    c.leg(0.68, 0.190, 308.9, nu_start_deg=180.0)
    c.leg(0.68, 0.398, 322.2, nu_start_deg=37.8, tilt_deg=2.2, n=280)
    c.coast_to(0.70, 0.650, 1.0, tilt_deg=-1.5)
    c.coast_to(0.68, 0.870, 5.2, tilt_deg=-0.8)
    c.coast_to(1.00, 0.802, 9.0, tilt_deg=1.5, n=220)
    c.aim_at([(1, "Venus", julian_day(1998, 4, 26)),
              (2, "Venus", julian_day(1999, 6, 24)),
              (3, "Earth", julian_day(1999, 8, 18)),
              (4, "Jupiter", julian_day(2000, 12, 30)),
              (5, "Saturn", julian_day(2004, 7, 1))])
    add("Cassini", (0.50, 0.42, 0.14), c)

    # --- Juno: launched Aug 2011 too slow to reach Jupiter, onto a two-year
    # ellipse out to 2.27 AU that brought it back past Earth in Oct 2013 for
    # the assist it actually needed.  That near-closed loop is the signature of
    # the trajectory and the reason Juno spent five years covering five AU; it
    # is drawn closed, which is a fraction of a revolution tidier than the real
    # one, whose two deep-space burns at aphelion left it slightly off.  Jupiter orbit insertion Jul 2016, close to
    # Jupiter's own aphelion at 5.45 AU.  The 53-day polar capture orbit around
    # Jupiter is smaller than the planet is drawn here, so the path ends on
    # arrival.
    c = Cruise(1.0, -20.0)
    c.revolution(0.98, 0.397, n=300)
    c.coast_to(0.98, 0.698, 5.45, tilt_deg=2.0, n=220)
    c.aim_at([(1, "Earth", julian_day(2013, 10, 9)),
              (2, "Jupiter", julian_day(2016, 7, 5))])
    add("Juno", (0.24, 0.34, 0.48), c)

    # --- Parker Solar Probe: a stack of nested ellipses, not a spiral inward.
    # It launched Aug 2018 onto an orbit that drops it straight down to Venus
    # seven weeks later, and each of the seven Venus assists since has shaved
    # energy off the orbit -- pulling perihelion in from 35.7 solar radii to
    # 9.86 while aphelion walks down from near Earth's orbit onto Venus's.
    # Every one of those seven orbits is a CLOSED ellipse, flown over and over
    # -- the final 88-day one comes back round every three months -- with
    # perihelion and aphelion on OPPOSITE sides of the Sun, 180 deg apart; one
    # full revolution of each is drawn.  Chaining them at the Venus crossing
    # makes them share that point and fans the line of apsides round by about
    # 30 deg across the set, the way the real one has moved.  The innermost
    # passes run inside the Sun as drawn here, which is a statement about the
    # drawn Sun being far larger than to scale rather than about the orbit:
    # 9.86 solar radii is 6.7 times closer in than Mercury ever gets, crossed
    # at 690,000 km/h -- the fastest anything built has ever moved.
    c = Cruise(1.0, 200.0)
    c.leg(0.45, 0.379, 248.0, nu_start_deg=180.0, n=140)
    for i, (q_rsun, r_aph) in enumerate(PSP_ORBITS):
        q = q_rsun * R_SUN_AU
        c.revolution(q, (r_aph - q) / (r_aph + q), tilt_deg=3.4 if i == 0 else 0.0, n=320)
    c.aim_at([(1, "Venus", julian_day(2018, 10, 3))])
    add("Parker Solar Probe", (0.55, 0.17, 0.09), c)

    return out


def preset_solar_system(rng, epoch_jd=None):
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

    WHERE each body sits on its orbit depends on epoch_jd.  Left None, every
    body is dropped at a random true anomaly: the orbits are real, the
    arrangement is not.  Given a Julian day, each is instead placed where it
    really is on that date, from its mean longitude (see "putting the planets
    where they really are" above) -- so the scene becomes a picture of the
    actual sky, and running it forward from there is a forecast rather than a
    doodle.  Both modes hand the integrator the same kind of state, so nothing
    downstream knows or cares which one built it.

    Moons and ring dust are NOT part of the N-body gravity pass -- a global
    timestep tuned for year-long planetary orbits is far too coarse to
    integrate a day-long moon orbit (let alone a ring particle's faster one)
    without it flying apart or aliasing into a strobing mess.  Instead they
    are kinematic satellites (see the "kinematic satellites" Taichi section):
    parented to their planet's live simulated position every frame, and swept
    round it on a circle whose PLANE, PERIOD and PHASE are the real ones,
    measured off JPL's ephemeris (the per-moon residuals are listed beside
    SATURN_EQUATOR_INCL above).  So on a given date each
    moon is on the correct side of its planet, going the correct way round, at
    the correct speed -- Io and Europa where they really are in the Galilean
    dance, Uranus's moons wheeling almost perpendicular to everything else
    because Uranus is tipped over, Triton and Charon running backwards.

    The one thing that is not real is how FAR out they are drawn.  Jupiter is
    drawn with a radius sixty times the size of Io's whole orbit, so a moon
    placed to scale would be buried inside its own planet; the radii are
    instead spread to a legible spacing that keeps the real ordering.  A moon
    is therefore in the right direction from its planet, at the wrong distance
    -- the same bargain the drawn planet radii themselves already make.
    Saturn's rings are the exception, and are drawn at their true 1.55 to 2.30
    Saturn radii.

    Two consequences worth knowing.  Phobos comes out orbiting Mars three
    times a day, faster than Mars turns, which is why it really does rise in
    the west; nothing here had to be told that, it falls out of a 7.65-hour
    period.  And at the speeds the panel offers, an inner moon completes
    thousands of orbits a second and can only alias -- moon motion is
    legible at a few days per second, and exact in real time.
    """
    G = 1.0
    m_sun = 1.0
    P, V, M, S, C, R = [], [], [], [], [], []
    labels = [(0, "Sun")]

    P.append([0.0, 0.0, 0.0]); V.append([0.0, 0.0, 0.0])
    # Softening at the floor, NOT at the Sun's drawn radius.  k_accel softens
    # with the source's own length, a = GM r / (r^2 + eps^2)^1.5, so a solar
    # eps of 0.6 units (0.06 AU) is a real weakening of the Sun's pull where
    # the planets actually are: 0.5% at Earth and 3.5% at Mercury.  The bodies
    # are launched from kepler_state at the UNSOFTENED speed, so that deficit
    # goes straight into the period -- Mercury came back 53 degrees short after
    # one of its own years, Earth 4 degrees -- and a scene whose whole claim is
    # that it shows a real date cannot afford either.  Nothing needs the
    # softening: the closest anything gravitating comes to the Sun is Mercury's
    # perihelion at 3.1 units, and the moons and rings are kinematic.
    M.append(m_sun); S.append(0.01); C.append([4.0, 3.1, 1.8]); R.append(2.6)

    moon_specs = []   # (parent_idx, planet_rad) + the moon's row from the table
    saturn_idx = saturn_rad = None
    orbit_lines = []
    for (name, a_au, e, i_deg, node_deg, peri_deg, l0_deg, dl_cy,
         m, rad, c, moons) in SOLAR_BODIES:
        a = a_au * AU_UNITS
        argp_deg = peri_deg - node_deg
        if epoch_jd is None:
            nu0 = rng.uniform(0.0, 2.0 * math.pi)
        else:
            nu0 = true_anomaly_at(epoch_jd, e, peri_deg, l0_deg, dl_cy)
        p_vec, v_vec = kepler_state(a, e, i_deg, node_deg, argp_deg, nu0, gm=G * m_sun)
        P.append(p_vec.tolist()); V.append(v_vec.tolist())
        M.append(m); S.append(0.25)
        C.append([c[0] * 1.6, c[1] * 1.6, c[2] * 1.6]); R.append(rad)
        parent_idx = len(P) - 1
        labels.append((parent_idx, name))
        orbit_lines.append({"name": name, "kind": "orbit",
                            "color": (c[0] * 0.55, c[1] * 0.55, c[2] * 0.55),
                            "pos": kepler_orbit_outline(a, e, i_deg, node_deg, argp_deg)})
        for moon in moons:
            moon_specs.append((parent_idx, rad) + tuple(moon))
        if name == "Saturn":
            saturn_idx, saturn_rad = parent_idx, rad

    n_src = len(P)

    def belt(count, r0, r1, thick, tint, rad, ecc=0.02):
        u = rng.uniform(0.0, 1.0, count)
        r = np.sqrt(r0 * r0 + u * (r1 * r1 - r0 * r0))
        ph = rng.uniform(0, 2 * math.pi, count)
        vc = np.sqrt(G * m_sun / r) * (1.0 + rng.normal(0, ecc, count))
        # Vertical state at a random PHASE of the oscillation rather than at
        # rest above the plane.  A Keplerian disk's vertical frequency is its
        # orbital one, so a belt laid down at its turning points with no
        # vertical motion has every particle at a given radius cross the
        # midplane at the same instant: the belt collapses to a razor-thin
        # sheet a quarter of an orbit in, puffs back out, and goes on doing it.
        # Drawing an amplitude and a phase independently gives a slab of
        # steady thickness instead -- the same rms height, in equilibrium.
        omega = np.sqrt(G * m_sun / r) / r
        amp = rng.normal(0.0, thick * math.sqrt(2.0), count)
        zph = rng.uniform(0, 2 * math.pi, count)
        P.extend(np.stack([r * np.cos(ph), amp * np.cos(zph),
                           r * np.sin(ph)], axis=1).tolist())
        V.extend(np.stack([-vc * np.sin(ph),
                           -amp * omega * np.sin(zph),
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

    def orbit_rate(period_days):
        """Angular rate in radians per SIM time unit, from a period in days.
        Fixed by the scene's units -- see YEAR_UNITS -- with nothing left to
        tune, which is what lets a moon keep the real time it keeps."""
        return 2.0 * math.pi / period_days * SIM_YEAR_DAYS / YEAR_UNITS

    def add_moon(parent_idx, planet_rad, name, radius_mult, size, color,
                 period_days, incl_deg, node_deg, u0_deg):
        u_ax, v_ax = orbit_plane_axes(incl_deg, node_deg)
        P.append([0.0, 0.0, 0.0]); V.append([0.0, 0.0, 0.0])
        M.append(0.0); S.append(0.03)
        C.append([color[0] * 1.9, color[1] * 1.9, color[2] * 1.9]); R.append(size)
        sat_parent.append(parent_idx)
        sat_u.append(u_ax); sat_v.append(v_ax)
        sat_radius.append(radius_mult * planet_rad)
        sat_rate.append(orbit_rate(period_days))
        # Dated, the moon starts at the argument of latitude it really has on
        # the day: its J2000 value carried forward at its own rate.  Undated,
        # there is no day for it to be the phase OF, so it gets a random one
        # exactly as the planets do.
        if epoch_jd is None:
            sat_phase0.append(rng.uniform(0, 2 * math.pi))
        else:
            turns = math.radians(u0_deg) + 2.0 * math.pi / period_days * (epoch_jd - J2000_JD)
            # Wrapped before it is stored, because the field is float32: a
            # century of Phobos is a million radians, and a million radians in
            # float32 has lost the tenth of a degree the wrap keeps.
            sat_phase0.append(math.fmod(turns, 2.0 * math.pi))

    for spec in moon_specs:
        add_moon(*spec)

    def saturn_ring(parent_idx, planet_rad, count):
        """Many independently-phased kinematic tracers rather than a static
        disk, so the ring visibly differentially-rotates the way a real one
        does -- inner material laps the outer edge many times over.

        The one part of this scene that IS drawn to scale: the rings really do
        run from 1.55 to 2.30 Saturn radii, so a ring particle's drawn radius
        is its true one and it can be given its true Keplerian period -- eight
        hours at the inner edge, fifteen at the outer.  They lie in Saturn's
        equatorial plane, the same plane its moons were just placed in, which
        is what stops the rings and Titan disagreeing about where Saturn's
        equator is."""
        r0, r1 = 1.55 * planet_rad, 2.30 * planet_rad
        # The Cassini division: 117,580-122,170 km, which is 1.951-2.027
        # Saturn radii, so it lands where the real gap is rather than near it.
        gap_lo, gap_hi = 1.951 * planet_rad, 2.027 * planet_rad
        r = r0 + rng.uniform(0.0, 1.0, count) * (r1 - r0)
        r = r[~((r > gap_lo) & (r < gap_hi))]
        n = r.shape[0]
        # Period of a circular orbit one Saturn radius out: 2 pi sqrt(R^3/GM)
        # with R = 60268 km and GM = 3.7931e7 km^3/s^2.  Every other radius
        # follows by Kepler's third law, which is the differential rotation.
        rate = orbit_rate(0.174706 * (r / planet_rad) ** 1.5)
        u_ax, v_ax = orbit_plane_axes(SATURN_EQUATOR_INCL, SATURN_EQUATOR_NODE)
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

    saturn_ring(saturn_idx, saturn_rad, 4000)

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

    # --- handedness ---------------------------------------------------------
    # Everything above is assembled in a frame where a prograde orbit runs from
    # +x toward +z -- the sense the belts, the rings and the mission arcs were
    # all written to agree with.  Seen from sim +y, which is the ecliptic north
    # pole and where the default camera sits, that comes out CLOCKWISE, and the
    # real Solar System seen from ecliptic north goes the other way: as built,
    # the scene is a mirror image of the sky.
    #
    # It was unnoticeable while every planet sat at a random longitude -- a
    # mirrored random arrangement is just another random arrangement.  On a
    # real date it is the difference between the picture and the sky, so the
    # scene is reflected through x here: one flip, applied to every position,
    # velocity, satellite axis and drawn line at once, so every internal
    # agreement made above survives it untouched, while the two reflections
    # compose into a plain rotation -- ecliptic north still at +y, and the
    # planets now going round it the way they really do.
    P = np.asarray(P, dtype=np.float64); P[:, 0] *= -1.0
    V = np.asarray(V, dtype=np.float64); V[:, 0] *= -1.0
    sat_u_arr, sat_v_arr = np.array(sat_u), np.array(sat_v)
    if sat_u_arr.size:
        sat_u_arr[:, 0] *= -1.0
        sat_v_arr[:, 0] *= -1.0
    satellites = (sat_parent, sat_u_arr, sat_v_arr, sat_radius, sat_rate, sat_phase0)
    for pl in polylines:
        pl["pos"] = pl["pos"] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)

    return Scene("Solar System", P, V, M, S, C, R,
                 n_src=n_src, n_dynamic=n_dynamic, satellites=satellites,
                 labels=labels, polylines=polylines, extras=extras,
                 gconst=G, dt=0.06, substeps=3, epoch_jd=epoch_jd,
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
        "gm": gm,
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
    velocity components come straight from the energy and angular momentum of
    the intended orbit -- in the Paczynski-Wiita potential the hole is actually
    integrated with, so the star's pericentre really is the one asked for."""
    m_star, r_star = cfg["m_star"], cfg["r_star"]
    r_apo, r_peri, gm = cfg["r_apo"], cfg["r_peri"], cfg["gm"]

    # The ball is built cold and centrally concentrated: a dense core survives
    # a little longer than its envelope, which is what puts the kink in the
    # stream and keeps the two tidal tails clearly separate rather than letting
    # the whole thing smear into one even fan.
    p, v = plummer_sphere(n, m_star, r_star * STAR_PLUMMER_A, 1.0, rng, rcut=2.4)

    # Re-centre the draw before it is placed.  A finite sample of a Plummer
    # sphere has its own centre of mass a little off the origin and drifting a
    # little, by order sigma/sqrt(N) -- and for a compact star sigma is
    # comparable to the orbital speed itself, so that sampling noise was worth
    # several percent of the orbit.  The pericentre then came out scattered
    # either side of the one asked for (15.9 to 17.9 r_s across three star
    # radii) for no reason but the seed.  Subtracting the sample's own mean
    # state puts the body's centre of mass exactly on the intended orbit and
    # leaves the ball otherwise untouched.
    n_src_c = max(1, min(int(n_src or n), n))
    p -= p[:n_src_c].mean(axis=0)
    v -= v[:n_src_c].mean(axis=0)

    r0 = float(np.clip(cfg.get("r_start", r_apo), r_peri * 1.6, r_apo))
    rh = float(cfg.get("rh", 0.0))
    if rh > 0.0:
        # The hole is integrated with the Paczynski-Wiita potential,
        # Phi = -GM/(r - r_s), so the energy and angular momentum that put the
        # apsides at r_peri and r_apo are that potential's, not Newton's.
        # Solving the Newtonian vis-viva instead -- which is what this did --
        # left the star measurably off the orbit it was advertised as being on:
        # with the defaults it reached 15.7 r_s rather than the 17 the panel
        # was computing beta against.
        # (now the spinning hole's pseudo-Newtonian potential, see k_accel)
        beta = float(cfg["beta"])
        phi_a = float(bhp.abn_potential(r_apo, gm, rh, beta))
        phi_p = float(bhp.abn_potential(r_peri, gm, rh, beta))
        denom = 1.0 / (r_peri * r_peri) - 1.0 / (r_apo * r_apo)
        l2 = 2.0 * (phi_a - phi_p) / max(denom, 1e-30)
        en = 0.5 * l2 / (r_apo * r_apo) + phi_a
        l_orb = math.sqrt(max(l2, 0.0))
        v0 = math.sqrt(max(2.0 * (en - float(bhp.abn_potential(r0, gm, rh, beta))), 0.0))
    else:
        a_orb = 0.5 * (r_apo + r_peri)
        ecc = (r_apo - r_peri) / (r_apo + r_peri)
        l_orb = math.sqrt(max(gm * a_orb * (1.0 - ecc * ecc), 0.0))
        v0 = math.sqrt(max(gm * (2.0 / r0 - 1.0 / a_orb), 0.0))
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
    # Softening for the massive subset: the spacing between them in the star's
    # core, so they model a smooth potential rather than scattering off one
    # another.  Derived from the ball rather than fixed, because the panel lets
    # the star radius go down to 0.5 r_s -- where a scale radius of 0.21 sits
    # INSIDE a fixed 0.30 softening, the ball has no self-gravity left to hold
    # it together, and it dissolves for numerical reasons while the panel
    # reports it surviving the pass.  At the default star this evaluates to
    # 0.30, which is the number it replaces.
    s[:n_src] = cfg.get("soft_src",
                        r_star * STAR_PLUMMER_A / max(n_src, 1) ** (1.0 / 3.0))
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

# Scenes that can be built for a real calendar date, and so take a Julian day
# as well as the RNG.  Only one has an ephemeris to be placed from: a galaxy
# merger or a collapsing cluster has no date to be at.
DATED_PRESETS = {1}


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
uniform float u_a;           // Kerr a = a* M in r_s (M = 1/2); + turns with the disk
uniform float u_rH;          // outer horizon, r_s
uniform int u_blackbody;     // 1 = disk coloured as a blackbody at its temperature
uniform float u_logTmax;     // log10 of the disk's peak temperature, K
uniform float u_logTstar;    // log10 surface temperature of the star, K
uniform float u_logTfloor;   // coolest disk colour shown (orange), log10 K
uniform sampler2D u_bbLUT;   // blackbody colour (unit luminance) over log10 T
uniform vec2 u_bbLogT;       // log10 T at the first and last texel
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
// fraction of the gas here that has joined the disk
float gasDisk(vec3 gc) {
    if (any(lessThan(gc, vec3(0.0))) || any(greaterThan(gc, vec3(1.0)))) return 0.0;
    return texture(u_gas, gc).g;
}

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
vec3 gasColor(float g, float r, float s);

// Cheap blackbody-ish ramp, t = 0 (cool outer disk) .. 1+ (inner, doppler boosted)
vec3 diskColor(float t) {
    t = clamp(t, 0.0, 1.6);
    vec3 c;
    if (t < 0.5) c = mix(vec3(0.55, 0.11, 0.02), vec3(1.00, 0.42, 0.06), t / 0.5);
    else if (t < 1.0) c = mix(vec3(1.00, 0.42, 0.06), vec3(1.00, 0.86, 0.54), (t - 0.5) / 0.5);
    else c = mix(vec3(1.00, 0.86, 0.54), vec3(0.86, 0.93, 1.10), (t - 1.0) / 0.6);
    return c;
}

// The visible colour of a blackbody at 10^lt K -- Planck's law through the
// CIE 1931 eye (bhp.blackbody_lut) -- normalised to unit luminance, so it
// sets the hue and the brightness is left to the emission model.
vec3 blackbodyColor(float lt) {
    float u = clamp((lt - u_bbLogT.x) / (u_bbLogT.y - u_bbLogT.x), 0.0, 1.0);
    return texture(u_bbLUT, vec2((u * 511.0 + 0.5) / 512.0, 0.5)).rgb;
}

vec3 gasColor(float g, float r, float s) {
    if (u_gasMode == 1) {
        vec3 c = mix(vec3(0.12, 0.16, 0.42), vec3(0.72, 0.34, 0.52),
                     smoothstep(0.02, 0.30, g));
        return mix(c, vec3(1.00, 0.90, 0.74), smoothstep(0.28, 0.75, g));
    }
    if (u_blackbody == 1) {
        // The star, and the stream it is drawn out into, radiate at the
        // star's own surface temperature: tidal stretching does not heat the
        // gas.  The fraction s of the gas here that has settled onto a disk
        // orbit (tracked per particle) has the disk's temperature for its
        // radius, T ~ r^-3/4.
        float x = clamp(u_diskIn / max(r, 1.2), 0.0, 1.0);
        float lt_disk = max(u_logTmax + 0.75 * log(x) / log(10.0) + 0.12, u_logTfloor);
        return blackbodyColor(mix(u_logTstar, lt_disk, s));
    }
    float t = pow(clamp(u_diskIn / max(r, 1.2), 0.02, 2.0), 0.75) + 0.92 * g;
    return diskColor(t);
}

// --- Kerr spacetime ---------------------------------------------------------
// OUTGOING Kerr-Schild Cartesian coordinates, spin axis Z.  World (x, y, z)
// maps to (x, z, -y), so Z is the disk's angular momentum (-y in the world)
// and a > 0 is a hole turning with the disk.  Outgoing because rays are traced
// BACKWARDS from the camera, and a ray traced back into the hole meets the
// past horizon, which these coordinates -- unlike the ingoing ones -- are
// regular across.  Checked against the exact Kerr critical impact parameters
// (a = 0: 3 sqrt 3 M; a = 0.9: 2.844 M / 6.832 M) to 4 figures.
//     g = eta + f k k,   f = 2 M r^3 / (r^4 + a^2 Z^2),
//     k = (1, -(r X - a Y)/(r^2 + a^2), -(r Y + a X)/(r^2 + a^2), -Z / r)
const float BH_M = 0.5;
vec3 toKS(vec3 w) { return vec3(w.x, w.z, -w.y); }
vec3 fromKS(vec3 k) { return vec3(k.x, -k.z, k.y); }

float ksR(vec3 x) {
    float a2 = u_a * u_a;
    float w = dot(x, x) - a2;
    return sqrt(max(0.5 * (w + sqrt(w * w + 4.0 * a2 * x.z * x.z)), 1e-12));
}

void ksMetric(vec3 x, out float f, out vec3 kv, out float r) {
    float a = u_a, a2 = a * a;
    float w = dot(x, x) - a2;
    float r2 = max(0.5 * (w + sqrt(w * w + 4.0 * a2 * x.z * x.z)), 1e-12);
    r = sqrt(r2);
    float D = r2 + a2;
    kv = -vec3((r * x.x - a * x.y) / D, (r * x.y + a * x.x) / D, x.z / r);
    f = 2.0 * BH_M * r * r2 / (r2 * r2 + a2 * x.z * x.z);
}

// Hamilton's equations for the (past-directed, p_t = +1) photon momentum:
//     H = 1/2 (|p|^2 - 1 - f (1 + l.p)^2),  l = -kv
void geo(vec3 x, vec3 p, out vec3 dx, out vec3 dp) {
    float a = -u_a, a2 = a * a;
    float w = dot(x, x) - a2;
    float S = sqrt(w * w + 4.0 * a2 * x.z * x.z);
    float r2 = max(0.5 * (w + S), 1e-12);
    float r = sqrt(r2);
    float D = r2 + a2;
    vec3 l = vec3((r * x.x + a * x.y) / D, (r * x.y - a * x.x) / D, x.z / r);
    float den = r2 * r2 + a2 * x.z * x.z;
    float f = 2.0 * BH_M * r * r2 / den;
    float Lq = dot(l, p) + 1.0;
    dx = p - f * Lq * l;
    vec3 gr = vec3(r2 * x.x, r2 * x.y, (r2 + a2) * x.z) / (r * max(S, 1e-9));
    vec3 gf = f * (3.0 * gr / r - (4.0 * r * r2 * gr + vec3(0.0, 0.0, 2.0 * a2 * x.z)) / den);
    float sxy = x.x * p.x + x.y * p.y;
    float A = r * sxy + a * (x.y * p.x - x.x * p.y);
    vec3 gA = gr * sxy + r * vec3(p.x, p.y, 0.0) + a * vec3(-p.y, p.x, 0.0);
    vec3 glp = gA / D - A * 2.0 * r * gr / (D * D) + vec3(0.0, 0.0, p.z) / r
             - x.z * p.z * gr / r2;
    dp = 0.5 * Lq * Lq * gf + f * Lq * glp;
}

float gdot4(vec4 u, vec4 v, float f, vec3 kv) {
    float ku = u.x + dot(kv, u.yzw);
    float kw = v.x + dot(kv, v.yzw);
    return -u.x * v.x + dot(u.yzw, v.yzw) + f * ku * kw;
}

vec3 nullify(vec3 x, vec3 p) {
    float f, r; vec3 kv; ksMetric(x, f, kv, r);
    vec3 d = normalize(p);
    float s = dot(kv, d);
    float A = 1.0 - f * s * s, B = 2.0 * f * s, C = -(1.0 + f);
    return d * ((-B + sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A));
}

// prograde circular orbit, Boyer-Lindquist radius rc (r_s), per r_s/c
float keplerOmega(float rc) {
    return sqrt(BH_M) / (pow(max(rc, 0.05), 1.5) + u_a * sqrt(BH_M));
}

// nu_obs / nu_emit for gas orbiting at om: gravitational redshift, Doppler
// shift and frame dragging in one exact expression (E = 1, L_z conserved).
float orbitG(vec3 x, float om, float Lz, float gCam) {
    float f, r; vec3 kv; ksMetric(x, f, kv, r);
    float rho2 = x.x * x.x + x.y * x.y;
    float lphi = -u_a * rho2 / (r * r + u_a * u_a);
    float nrm = -((-1.0 + f) + 2.0 * om * f * lphi + om * om * (rho2 + f * lphi * lphi));
    if (nrm <= 1e-6) return 0.0;
    return gCam * sqrt(nrm) / max(1.0 - om * Lz, 1e-4);
}

// Boyer-Lindquist cylindrical radius of a world-axes point (hole-centred, r_s)
float blCyl(vec3 w) {
    vec3 x = toKS(w);
    float r = ksR(x);
    float ct = clamp(x.z / r, -1.0, 1.0);
    return r * sqrt(1.0 - ct * ct);
}

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 rd = normalize(u_camFwd + u_camRight * ndc.x * u_tanHalf * u_aspect
                                 + u_camUp * ndc.y * u_tanHalf);

    // Work in Schwarzschild radii with the hole at the origin.
    vec3 p = (u_camPos - u_bhPos) / u_rs;
    vec3 v = rd;

    // The photon arriving at this pixel, as seen by a static camera: build
    // its orthonormal frame in the Kerr metric (project the axes off its
    // 4-velocity, Gram-Schmidt), send the photon in along -n, and keep its
    // past-directed momentum P (p_t = +1) in Kerr-Schild coordinates X.  p and
    // v below stay the world-axes position and direction every volume is
    // sampled with.
    vec3 X = toKS(p);
    vec3 P;
    float gCam;
    {
        vec3 n = toKS(rd);
        float f, r; vec3 kv; ksMetric(X, f, kv, r);
        vec4 u = vec4(inversesqrt(max(1.0 - f, 1e-4)), 0.0, 0.0, 0.0);
        vec4 e0 = vec4(0.0, 1.0, 0.0, 0.0);
        e0 += gdot4(u, e0, f, kv) * u;
        e0 *= inversesqrt(gdot4(e0, e0, f, kv));
        vec4 e1 = vec4(0.0, 0.0, 1.0, 0.0);
        e1 += gdot4(u, e1, f, kv) * u;
        e1 -= gdot4(e0, e1, f, kv) * e0;
        e1 *= inversesqrt(gdot4(e1, e1, f, kv));
        vec4 e2 = vec4(0.0, 0.0, 0.0, 1.0);
        e2 += gdot4(u, e2, f, kv) * u;
        e2 -= gdot4(e0, e2, f, kv) * e0;
        e2 -= gdot4(e1, e2, f, kv) * e1;
        e2 *= inversesqrt(gdot4(e2, e2, f, kv));
        vec4 k = u - (n.x * e0 + n.y * e1 + n.z * e2);
        float lk = k.x + dot(kv, k.yzw);
        float E = k.x - f * lk;
        P = -(k.yzw + f * lk * kv) / E;
        gCam = 1.0 / E;
    }

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
            X = toKS(p);
            P = nullify(X, P);
        } else {
            steps = 0;            // path's closest approach never reaches enterR
        }
    }

    float r0 = length(p);
    float Lz = X.y * P.x - X.x * P.y;    // photon's angular momentum, conserved
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

    vec3 dX, dP;
    geo(X, P, dX, dP);

    for (int i = 0; i < steps; i++) {
        float r = ksR(X);
        if (r < u_rH * 1.01) { captured = true; break; }
        if (r > escR && dot(p, v) > 0.0) break;
        if (trans < 0.004) break;

        // The cap used to be 1.3 everywhere, which is far finer than empty
        // space needs and is what made a long beam or a wide debris field run
        // the step budget out before the ray got there.
        float dt = clamp(0.11 * (r - 0.92 * u_rH), 0.012, 3.5);
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

        // RK4 along the Kerr null geodesic
        vec3 pPrev = p;
        vec3 k2x, k2p, k3x, k3p, k4x, k4p;
        geo(X + 0.5 * dt * dX, P + 0.5 * dt * dP, k2x, k2p);
        geo(X + 0.5 * dt * k2x, P + 0.5 * dt * k2p, k3x, k3p);
        geo(X + dt * k3x, P + dt * k3p, k4x, k4p);
        X += dt / 6.0 * (dX + 2.0 * k2x + 2.0 * k3x + k4x);
        P += dt / 6.0 * (dP + 2.0 * k2p + 2.0 * k3p + k4p);
        geo(X, P, dX, dP);
        p = fromKS(X);
        v = normalize(fromKS(dX));

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
            // Boyer-Lindquist radius, the one the ISCO (u_diskIn) is quoted in
            float rr = blCyl(x);
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
                // Kerr orbital angular velocity; the disk turns -phi about +y
                float om = -keplerOmega(rr);
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

                // Novikov-Thorne: the flux of a thin disk goes as
                // r^-3 (1 - sqrt(r_in / r)) -- zero at the ISCO, where no
                // torque holds the gas, peaking at (49/36) r_in, then falling
                // as r^-3.  Normalised to 1 at that peak.  Faded out over the
                // outer half so the disk's edge is gas thinning, not a cut.
                float xi = u_diskIn / rr;
                float emis = xi * xi * xi * max(1.0 - sqrt(xi), 0.0) / 0.05665;
                float radial = 1.0 - smoothstep(0.45, 1.0, tn);

                // Exact Kerr redshift of gas on a circular orbit here:
                // Doppler, gravitational redshift and frame dragging together
                // (this used to be the Schwarzschild special case).
                float shift = clamp(orbitG(toKS(x), keplerOmega(rr), Lz, gCam), 0.05, 3.2);

                float boost = shift * shift * shift;
                vec3 dcol;
                if (u_blackbody == 1) {
                    // Blackbody.  The local temperature follows the flux,
                    // sigma T^4 ~ emis (the Novikov-Thorne profile, 1 at its
                    // peak), so T = T_max emis^(1/4); what reaches the camera
                    // is a blackbody at g T (I_nu / nu^3 is invariant), so the
                    // approaching side is bluer as well as brighter.
                    float lt = max(u_logTmax + 0.25 * log(max(emis, 1e-8)) / log(10.0),
                                   u_logTfloor)
                             + log(shift) / log(10.0);
                    dcol = blackbodyColor(lt);
                } else {
                    dcol = diskColor(pow(u_diskIn / rr, 0.75) * shift);
                }
                vec3 emit = dcol * emis * lanes * radial * boost * u_diskBright;

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
                    col += trans * gasColor(g, length(mid - u_gasOff),
                                            gasDisk(gasCoord(mid)))
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
        self.bh_spin = 0.9        # the hole's a*, set from Sim.spin every frame
        self.blackbody = True     # disk colour from its real temperature
        # Show the disk with its peak temperature at DISK_TPEAK_SHOWN instead
        # of the real ~10^7 K.  The profile (T ~ r^-3/4 with the zero-torque
        # inner edge) and the Doppler/gravitational shifts are unchanged, so
        # the colour runs white -> yellow -> red outward exactly as a real
        # blackbody disk does when its inner edge is that hot -- as around a
        # supermassive hole.  At 10^7 K every ring is the same blue-white.
        self.disk_tscaled = True
        self.log_tmax = 7.0       # log10 peak disk temperature, set every frame
        self.log_tstar = math.log10(5772.0)   # the star's surface, set every frame

        self.bb_tex = ctx.texture((bhp.BB_N, 1), 4, bhp.blackbody_lut().tobytes(),
                                  dtype="f4")
        self.bb_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bb_tex.repeat_x = self.bb_tex.repeat_y = False
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
        self.gas_tex = ctx.texture3d((GAS_NX, GAS_NY, GAS_NZ), 2, dtype="f1")
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
        self.gas_tex.write(bytes(GAS_NX * GAS_NY * GAS_NZ * 2))

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
            setu(g, "u_a", 0.5 * self.bh_spin)
            setu(g, "u_rH", 0.5 * bhp.horizon(self.bh_spin))
            self.bb_tex.use(5)
            setu(g, "u_bbLUT", 5)
            setu(g, "u_bbLogT", (bhp.BB_LOGT_MIN, bhp.BB_LOGT_MAX))
            setu(g, "u_blackbody", 1 if self.blackbody else 0)
            setu(g, "u_logTmax", self.log_tmax)
            setu(g, "u_logTstar", self.log_tstar)
            setu(g, "u_logTfloor", math.log10(DISK_TMIN_SHOWN) if self.disk_tscaled else 0.0)
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
        # Julian day the Solar System is built for, or None for the original
        # random-longitude arrangement.  Held here rather than on the scene so
        # it survives a restart and a trip through the other scenes.
        self.epoch_jd = ARGS_EPOCH_JD
        # Run the clock at wall-clock speed -- one real second is one simulated
        # second -- instead of at dt * speed, which crosses a fortnight every
        # second and carries a scene built for "now" straight off the present.
        # Only means anything in a dated scene; there is no real time to keep
        # in one that is not at a date.
        self.real_time = ARGS_REAL_TIME
        self.wall_prev = None   # time.time() at the last frame the clock moved
        # Was the date asked for "now" rather than a particular day?  If it
        # was, the scene is stamped with the instant it is BUILT, not the one
        # the arguments were read at -- otherwise a live view opens already
        # behind by however long the shaders took to compile, and a restart
        # would put it back to the moment the program started rather than to
        # the present.
        self.epoch_is_now = ARGS_EPOCH_IS_NOW
        # tidal-disruption controls
        self.self_gravity = True
        # The hole's spin, J c / (G M^2).  + turns with the disk.  Sets the
        # horizon, ISCO, capture radius, the force the debris feels, the
        # light bending, and (no spin, no jet) the jets.
        self.spin = 0.9
        self.kerr = Kerr(self.spin)
        self.circ = 0.08            # circularisation: fraction of the stream's
                                    # radial motion its self-intersection shocks
                                    # remove per radian of orbit
        self.alpha = 0.1            # Shakura-Sunyaev viscosity; 0.01-0.3 from MRI
                                    # simulations and dwarf-nova outbursts.  Sets
                                    # how long the disk lasts: t_visc =
                                    # 1 / (alpha (H/R)^2 Omega)
        self.disk_h = 1.0           # kept in step with Renderer.disk_h (drawing only)
        # The disk's PHYSICAL state, measured from the debris every survey
        # (App.update_tde): its accretion rate, the thickness that rate gives
        # it, and the temperature of its hottest ring.  The viscosity runs on
        # this thickness, not on the drawn one -- how fast a disk drains goes
        # as (H/R)^2, and H/R is set by how hard it is being fed.
        self.hr_visc = bhp.ADVECTIVE_HR
        self.mdot = 0.0             # kg/s
        self.mdot_edd = 0.0         # in units of the Eddington rate
        self.t_max = 0.0            # K, peak effective temperature
        self.t_star = 5772.0        # K, surface of the star most recently dropped
        # Nothing is resupplied.  This used to re-inject everything the hole
        # swallowed back at a "feed radius", so the disk never emptied; a real
        # disk made from one star lasts exactly as long as that star's gas.
        self.sustain_disk = False
        self.feed_radius = 26.0     # where a sustained disk is resupplied
        self.return_radius = 210.0  # past here, scattered material is brought back
        self.star_intact = False    # self-gravity only matters while a star is whole
        self.jet = True             # twin polar jets off the inner disk
        self.jet_rate = 0.0018      # per-frame chance a disk particle is launched
        self.jet_range = 70.0       # jet material is returned to the disk past here
        self.jet_source = 18.0      # jets are fed from inside this radius
        self.jet_base = 2.2         # launch height above the hole
        # Launch speed, c = 1.  This has to beat escape in the potential the
        # material is integrated in, which is Paczynski-Wiita: v_esc =
        # sqrt(2GM/(r - r_s)) = 0.91 at the 2.2 r_s launch height, not the
        # Newtonian sqrt(2GM/r) = 0.67.  At 0.88 the beam was bound -- it
        # turned round at 18 r_s and rained back into the disk, while the
        # drawn jet went on to 70, so the two disagreed about how long the
        # jet was.  Above escape it coasts out to the return radius and is
        # recycled from there, which is the loop the code was written for.
        self.jet_speed = 0.95
        self.jet_spread = 0.032     # opening angle as a fraction of jet speed
        self.circularising = False  # switched on once the star reaches pericentre

    def set_spin(self, a):
        self.spin = bhp.clamp_spin(a)
        self.kerr = Kerr(self.spin)

    def hole_force(self):
        """(r_H, beta) of the spinning hole's force, or (0, 0) with no hole."""
        if self.scene is None or self.scene.pw_rs <= 0.0:
            return 0.0, 0.0
        return self.kerr.rh * self.scene.pw_rs, self.kerr.beta

    def load(self, key):
        self.key = key
        if self.epoch_is_now and self.epoch_jd is not None:
            self.epoch_jd = today_jd()
        if key in DATED_PRESETS:
            self.scene = PRESETS[key](self.rng, self.epoch_jd)
        else:
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
        k_accel(s.n_dynamic, lo, hi, s.gconst, *self.hole_force())
        self.time = 0.0
        # Re-anchor the wall clock too: a scene rebuilt at a new date must not
        # inherit a stopwatch that has been running since the last one.
        self.wall_prev = None
        self.circularising = False
        self.star_intact = False
        return s

    def live(self):
        """Is the clock following the wall clock this frame?  Real time needs a
        scene that has a date for it to mean anything."""
        return self.real_time and self.scene is not None and self.scene.epoch_jd is not None

    def hold_clock(self):
        """Let the wall clock run without the simulation following it -- what
        pausing has to do, so that resuming carries on from where it stopped
        rather than teleporting forward by the length of the pause."""
        self.wall_prev = time.time()

    def real_time_dt(self):
        """Sim time units to advance this frame so the run keeps pace with the
        wall clock, one second per second.  Clamped (see REALTIME_MAX_CATCHUP)
        and self-anchoring: the first frame after the mode is switched on
        advances nothing, it only starts the stopwatch."""
        now = time.time()
        prev, self.wall_prev = self.wall_prev, now
        if prev is None:
            return 0.0
        secs = min(max(now - prev, 0.0), REALTIME_MAX_CATCHUP)
        return secs / 86400.0 / SIM_YEAR_DAYS * YEAR_UNITS

    def current_jd(self):
        """The date the run has reached, as a Julian day, or None if the scene
        was not built for one.  Sim time is a real elapsed time in a scene
        whose units are pinned to the AU and the solar mass, so the clock is
        simply the starting date plus however far the integrator has walked --
        no separate bookkeeping, and it stays right through pausing, stepping
        and every change of speed."""
        s = self.scene
        if s is None or s.epoch_jd is None:
            return None
        return s.epoch_jd + self.time / YEAR_UNITS * SIM_YEAR_DAYS

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
        # the orbit is solved in the potential of the hole at its current spin
        s.star_cfg["rh"], s.star_cfg["beta"] = self.hole_force()
        p, v, m, soft = make_star(s.star_cfg, s.star_n, self.rng, azimuth,
                                  n_src=s.star_src)
        k_write_block(lo, s.star_n, p, v, m, soft)
        k_fill_in_disk(lo, lo + s.star_n, 0.0)   # a fresh star is star, not disk
        s.star_spawned += 1
        # grow the live range to cover every slot used so far; every dynamic
        # particle here is also a gravity source, so the star holds itself
        # together and its debris keeps pulling on itself
        live = 1 + min(s.star_spawned, s.star_slots) * s.star_n
        s.n_dynamic = max(s.n_dynamic, live)
        s.n_base = s.n_dynamic + s.n_sat
        self.star_intact = True
        s.star_lo = lo
        self.t_star = main_sequence_teff(s.star_cfg["m_star"] * SIM_MASS_TO_SOLAR)
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
        k_accel(s.n_dynamic, src_lo, src_hi, s.gconst, *self.hole_force())
        return True

    def step(self):
        s = self.scene
        if self.live():
            # A frame's worth of real time is a few millionths of a time unit
            # here, so the integrator is being asked for a far finer step than
            # it was tuned for rather than a coarser one: accurate, cheap, and
            # the planets move exactly as slowly as the real ones do.
            dt = self.real_time_dt() / max(s.substeps, 1)
            if dt <= 0.0:
                return
        else:
            dt = s.dt * self.speed
        # Self-gravity is only what holds a star together on the way in; once
        # it is a spread-out stream the hole dominates utterly, so the source
        # range collapses to the hole afterwards and the cost drops from
        # O(N^2) to O(N).  See Sim.src_bounds.
        src_lo, src_hi = self.src_bounds()
        s.n_src = src_hi - src_lo + (1 if s.pw_rs > 0.0 else 0)
        eating = bool(s.star_n) and s.n_dynamic > 1
        nsub = s.substeps
        if eating:
            # Enough substeps to resolve an orbit at the ISCO: prograde spin
            # brings it in to 0.62 r_s, where an orbit takes a sixth of the
            # time it does at 3 r_s.
            t_isco = 2.0 * math.pi / self.kerr.omega(self.kerr.isco)
            frame = dt * nsub
            nsub = min(max(nsub, int(math.ceil(frame / (t_isco / 80.0)))), 48)
            dt = frame / nsub
        rh, beta = self.hole_force()
        for _ in range(nsub):
            k_kick(s.n_dynamic, 0.5 * dt)
            k_drift(s.n_dynamic, dt)
            k_accel(s.n_dynamic, src_lo, src_hi, s.gconst, rh, beta)
            k_kick(s.n_dynamic, 0.5 * dt)
            self.time += dt
            if eating:
                s.accreted += k_swallow(1, s.n_dynamic, s.pw_rs,
                                        1 if self.sustain_disk else 0,
                                        s.gconst * s.mass[0], self.feed_radius,
                                        self.return_radius, self.jet_range,
                                        self.kerr.r_cap * s.pw_rs)
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
            if self.circularising:
                # H/R of the physical disk (see hr_visc)
                k_circularise(1, s.n_dynamic, dt * nsub, self.circ, self.alpha,
                              s.gconst * s.mass[0], rh, beta,
                              self.hr_visc, 0.0)
            # (The jets are drawn by the ray marcher.  Disk particles are no
            # longer teleported onto the axis to fake them: a jet carries a
            # negligible fraction of the accreted mass, and with nothing
            # resupplied every particle launched was a hole in the disk.)
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
        # Year/month/day the date box is showing.  Only handed to the sim when
        # Apply is pressed, so typing a year is not four scene rebuilds.
        self.date_ymd = list(calendar_date(sim.epoch_jd if sim.epoch_jd is not None
                                           else today_jd()))
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
        self.disk_cnt_peak = 1    # most debris ever on disk orbits this meal
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
        self.disk_cnt_peak = 1
        self.renderer.clear_gas()
        if self.scene.star_n:
            # a bare hole: nothing to light up until a star has been torn apart
            self.renderer.disk_bright = 0.0
            self.renderer.jet_bright = 0.0
            self.scene.disk_in = self.sim.kerr.isco * self.scene.pw_rs
            self.scene.disk_out = self.scene.disk_in + 1.0
        return self.scene

    def update_gas(self):
        """Re-splat the debris into the density grid the renderer marches."""
        s, r = self.scene, self.renderer
        if s.star_n and s.n_dynamic > 1 and self.sim.circularising:
            rh, beta = self.sim.hole_force()
            if self.sim.star_intact:
                core = tuple(float(c) for c in self.star_core)
                core_r2 = (1.5 * s.star_cfg["r_star"]) ** 2
            else:
                core, core_r2 = (GRAVEYARD, GRAVEYARD, GRAVEYARD), 0.0
            k_mark_disk(1, s.n_dynamic, s.gconst * s.mass[0], rh, beta, core, core_r2)
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
            # The hole has finished its meal.  This used to return, which
            # left the disk and the jets frozen at whatever brightness they
            # last had -- beams still blazing over an empty hole for the
            # rest of the run.  There is nothing left to accrete, so fall
            # through to the easing below with the targets at zero and let
            # them go out the same way they came in.
            self.debris_frac = 0.0
            self.tgt_disk = (0.0, s.disk_in, s.disk_out)
            self.star_bound = 0.0
            sim.star_intact = False
        else:
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
                rh_w, beta_w = sim.hole_force()
                v_c = bhp.abn_vcirc(np.maximum(r_alive, 1.01 * rh_w),
                                    s.gconst * s.mass[0], rh_w, beta_w)
                l_mag = np.linalg.norm(np.cross(p_alive, vel_all), axis=1)
                kappa = l_mag / np.maximum(r_alive * v_c, 1e-9)
                # no radius cap: the disk is wherever the circularised gas is
                disky = (v_r < 0.30 * v_c) & (kappa > 0.75) & (kappa < 1.3)
                if sim.star_intact:
                    # a star still whole on its way through pericentre is
                    # momentarily on a near-circular arc, but it is a star
                    disky &= (np.linalg.norm(p_alive - self.star_core, axis=1)
                              > 1.5 * s.star_cfg["r_star"])
                cnt = int(disky.sum())
                self.debris_frac = cnt / float(n_alive)

                self.tgt_disk = (0.0, s.disk_in, s.disk_out)
                if cnt > 60:
                    rr = r_alive[disky]
                    self.debris_inner = float(np.percentile(rr, 8))
                    # Outer edge: where the gas on circular orbits actually is.
                    # Only gas on near-circular orbits counts (the test above),
                    # so scattered eccentric material cannot inflate it; the
                    # 95th percentile drops the last few stragglers.  It starts
                    # near the circularisation radius, ~2 r_p, and moves as
                    # viscosity spreads the ring.
                    self.debris_outer = float(np.percentile(rr, 95))
                    # Inner edge: the ISCO of the hole's spin.  A thin disk
                    # radiates down to it and no further -- inside it gas
                    # plunges -- whatever the sparse particles there suggest.
                    tgt_in = sim.kerr.isco * s.pw_rs
                    # Brightness follows how much gas is left.  A disk's
                    # luminosity is its accretion rate, Mdot ~ M_disk / t_visc,
                    # so it fades as the ring drains -- rather than following
                    # the FRACTION of debris on disk orbits, which stays near 1
                    # as the last of it goes and kept the disk blazing.
                    self.disk_cnt_peak = max(self.disk_cnt_peak, cnt)
                    self.disk_physics(sim, s, rr, cnt * step)
                    self.tgt_disk = (self.disk_peak * cnt / self.disk_cnt_peak,
                                     tgt_in,
                                     max(self.debris_outer, tgt_in + 1.0))

        tgt_bright, tgt_in, tgt_out = self.tgt_disk
        if tgt_bright <= 0.0:
            sim.mdot = sim.mdot_edd = sim.t_max = 0.0
        if self.auto_disk:
            # ease toward the target so the disk grows in smoothly instead of
            # snapping around as debris sloshes through pericentre
            k = 0.02
            r.disk_bright += (tgt_bright - r.disk_bright) * k
            s.disk_in += (tgt_in - s.disk_in) * k
            s.disk_out += (tgt_out - s.disk_out) * k
        # The jets are powered by accretion, so they live and die with the
        # disk: once the hole has finished its meal there is nothing left to
        # launch and the beams must go out with it.  Scaling them on
        # debris_frac is not enough to do that, because debris_frac is a
        # FRACTION -- as the disk drains, the count of disk-like particles and
        # the count of surviving ones fall together, so the ratio can sit near
        # 1 with almost nothing left, and the beams stay blazing over an empty
        # hole.  The disk's own brightness already carries the absolute measure
        # (it is held at zero below the handful-of-particles floor above), so
        # drive the jets off that instead, and off the drawn brightness when
        # the disk is being flown by hand rather than grown from the debris.
        disk_now = tgt_bright if self.auto_disk else r.disk_bright
        tgt_jet = 0.0
        if sim.jet and sim.circularising and self.disk_peak > 1e-6:
            # Blandford-Znajek: the jet is the hole's spin energy extracted by
            # the field the disk brings in, power ~ Omega_H^2 -- so it scales
            # with spin (relative to a = 0.9, where it has its old brightness)
            # and there is no jet at all from a hole that does not spin.
            bz = bhp.bz_efficiency(sim.spin, 30.0) / bhp.bz_efficiency(0.9, 30.0)
            tgt_jet = (self.jet_peak * min(bz, 3.0)
                       * min(1.0, max(0.0, disk_now) / self.disk_peak))
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
        if r.blackbody:
            # the star's own blackbody until the gas is shocked in the disk,
            # the disk's temperature for its radius after (as the gas pass)
            lt_star = math.log10(sim.t_star)
            x = np.clip(s.disk_in / np.maximum(rad, 1.2), 1e-6, 1.0)
            lt_disk = r.log_tmax + 0.75 * np.log10(x) + 0.12
            if r.disk_tscaled:
                lt_disk = np.maximum(lt_disk, math.log10(DISK_TMIN_SHOWN))
            w = in_disk.to_numpy()[1:s.n_dynamic]
            lt = lt_star + (lt_disk - lt_star) * w
            col[:] = bhp.blackbody_rgb(10.0 ** lt) * boost[:, None]
        else:
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

    def disk_physics(self, sim, s, rr, n_disk):
        """Accretion rate, thickness and temperature of the disk the debris
        has formed, from how much gas is in it and where.

          Mdot  ~ M_disk / t_visc,  t_visc = 1 / (alpha (H/R)^2 Omega) at the
                  disk's median radius
          H/R   from that rate's Eddington ratio (bhp.scale_height): thin when
                  fed gently, radiation-pressure puffed up to ~0.5 when fed at
                  many times Eddington -- which a disrupted star always is
          T     peak of the Novikov-Thorne profile, sigma T^4 =
                  3 G M Mdot / (8 pi r^3) (1 - sqrt(r_in / r)) at r = 49/36 r_in,
                  capped at the Eddington flux the surface can radiate

        Iterated a few times because the rate and the thickness each set the
        other; it converges in two or three."""
        kr = sim.kerr
        cfg = s.star_cfg
        m_rep = cfg["m_star"] * SIM_MASS_TO_SOLAR * bhp.M_SUN_KG / max(s.star_n, 1)
        m_disk = n_disk * m_rep
        t_unit = BH_RS_KM * 1000.0 / bhp.C_SI              # s per r_s / c
        r_med = float(np.median(rr)) / s.pw_rs              # r_s
        om = kr.omega(max(r_med, kr.isco)) / t_unit          # 1/s
        m_bh = BH_SOLAR_MASSES * bhp.M_SUN_KG
        mdot_edd_kg = bhp.eddington_luminosity(BH_SOLAR_MASSES) / (kr.eta * bhp.C_SI ** 2)
        hr = sim.hr_visc
        for _ in range(4):
            mdot = m_disk * sim.alpha * hr * hr * om
            hr = float(bhp.scale_height(2.0 * r_med, kr.a, mdot / mdot_edd_kg, kr.eta))
        sim.hr_visc, sim.mdot, sim.mdot_edd = hr, mdot, mdot / mdot_edd_kg
        rs_m = BH_RS_KM * 1000.0
        r_pk = (49.0 / 36.0) * kr.isco * rs_m
        f_nt = (3.0 * bhp.G_SI * m_bh * mdot / (8.0 * math.pi * r_pk ** 3)
                * (1.0 - math.sqrt(36.0 / 49.0)))
        f_cap = float(bhp.eddington_flux(r_pk, m_bh, hr))
        sim.t_max = (min(f_nt, f_cap) / bhp.SIGMA_SB) ** 0.25

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


MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def draw_date_controls(app):
    """The date box: switch the Solar System between bodies at random
    longitudes and bodies placed where they really are on a chosen day, and
    then show the date the run has walked forward to."""
    sim, scene = app.sim, app.scene

    imgui.separator_text("date")
    dated = scene.epoch_jd is not None
    changed, val = imgui.checkbox("place bodies on a real date", dated)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "On: every planet and dwarf planet starts where it really is on"
            " the date below, from its published mean longitude -- the scene"
            " becomes the actual sky, and running it forward is a forecast."
            "\nOff: the same real orbits, but each body dropped at a random"
            " point along its own.")
    if changed:
        sim.epoch_jd = julian_day(*app.date_ymd) if val else None
        sim.epoch_is_now = False
        app.load(1)
        dated = val

    imgui.begin_disabled(not dated)
    imgui.push_item_width(-72)
    changed, vals = imgui.input_int3("y / m / d", app.date_ymd)
    if changed:
        y, m, d = vals
        m = max(1, min(12, int(m)))
        # Clamped against the month rather than rolled over, so holding the
        # stepper on the day field walks to the end of the month and stops
        # instead of silently changing which month is being asked for.
        app.date_ymd = [max(-4000, min(9999, int(y))), m,
                        max(1, min(MONTH_DAYS[m - 1], int(d)))]
    imgui.pop_item_width()

    if imgui.button("Apply", imgui.ImVec2(84, 0)):
        sim.epoch_jd = julian_day(*app.date_ymd)
        sim.epoch_is_now = False
        app.load(1)
    imgui.same_line()
    if imgui.button("J2000", imgui.ImVec2(84, 0)):
        app.date_ymd = [2000, 1, 1]
        sim.epoch_jd = J2000_JD
        sim.epoch_is_now = False
        app.load(1)
    imgui.end_disabled()

    # Left live even when the scene is not dated yet: "show me right now" is
    # the whole point of the box, and it should not first need the mode above
    # switching on by hand.
    imgui.same_line()
    if imgui.button("Now", imgui.ImVec2(84, 0)):
        # This instant, seconds and all, with the clock locked to the wall
        # clock -- asking to see now and then watching it race off the present
        # would undo the one thing this button is for.
        sim.epoch_jd = today_jd()
        sim.real_time = True
        sim.epoch_is_now = True
        app.date_ymd = list(calendar_date(sim.epoch_jd))
        app.load(1)
        dated = True

    imgui.begin_disabled(not dated)
    changed, val = imgui.checkbox("run at real time  (1 s = 1 s)", sim.real_time)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Advance the scene at wall-clock speed, so the date and time below"
            " are the real ones and the planets crawl exactly as slowly as"
            " they really do."
            "\nOff, the speed slider takes over and a second of watching is"
            " about a fortnight of Solar System.")
    if changed:
        sim.real_time = val
        sim.wall_prev = None
    imgui.end_disabled()

    # Read the clock off whatever scene is loaded NOW: any of the buttons above
    # may have rebuilt it a few lines ago, which leaves the local stale.
    jd, epoch = sim.current_jd(), sim.scene.epoch_jd
    if jd is None:
        imgui.text_disabled("bodies at random longitudes")
        return
    y, m, d = calendar_date(jd)
    hh, mm, ss = clock_time(jd)
    years = (jd - epoch) / SIM_YEAR_DAYS
    imgui.text(f"showing  {y:5d}-{m:02d}-{d:02d}  {hh:02d}:{mm:02d}:{ss:02d} UTC")
    if sim.live() and not sim.epoch_is_now:
        # Real time from a date that was never meant to be now: the rate is the
        # real one, the moment is not, so there is nothing to be behind.
        imgui.text_disabled(f"real time, {years:+.3f} yr from the start")
    elif sim.live():
        behind = (today_jd() - jd) * 86400.0
        # Anything more than a second or two adrift was a pause, a minimised
        # window or a sleeping machine -- say so, since the whole promise of
        # this mode is that the clock on screen is the real one.
        if abs(behind) < 2.0:
            imgui.text_disabled("live -- tracking the wall clock")
        else:
            imgui.text_disabled(f"{behind / 60.0:.1f} min behind now -- press Now")
    else:
        imgui.text(f"elapsed  {years:+9.3f} yr")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Sim time is a real elapsed time here: the scene's length unit"
                " is the AU and its mass unit the Sun, so one Earth orbit IS"
                " one year. Pause, step or change the speed and the date still"
                " follows the integrator.")
    # The published elements are a fit over 1800-2050; a linear mean longitude
    # keeps working outside it but the planets slide out of true, so say so
    # rather than let a year-3000 screenshot pass for an ephemeris.
    if not 1800 <= y <= 2050:
        imgui.text_disabled("outside 1800-2050: approximate")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The orbital elements are fitted over 1800-2050, where these"
                " positions sit within a thousandth of an AU of JPL's for the"
                " inner planets and a hundredth for the outer ones. Further"
                " out the drift grows and they become indicative rather than"
                " accurate.")


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

    # In real time the step is dictated by how long the last frame took, so
    # neither dial has anything left to say; they are greyed rather than hidden
    # so it is clear which control is holding them, and where to give them back.
    imgui.begin_disabled(sim.live())
    changed, val = imgui.slider_float("speed", sim.speed, 0.05, 16.0, "%.2fx",
                                      imgui.SliderFlags_.logarithmic)
    if changed:
        sim.speed = val
    imgui.end_disabled()
    changed, val = imgui.slider_int("substeps", scene.substeps, 1, 8)
    if changed:
        scene.substeps = val
    imgui.begin_disabled(sim.live())
    changed, val = imgui.slider_float("dt", scene.dt, 0.002, 0.4, "%.4f",
                                      imgui.SliderFlags_.logarithmic)
    if changed:
        scene.dt = val
    imgui.end_disabled()
    if sim.live():
        imgui.text_disabled("speed and dt set by the real-time clock")

    imgui.separator_text("state")
    imgui.text(f"scene       {scene.name}")
    # A frame of real time is a few millionths of a time unit, so the usual
    # two decimals would sit at 0.00 all session and read as a stalled clock.
    imgui.text(f"sim time  {sim.time:12.6f}" if sim.live()
               else f"sim time    {sim.time:10.2f}")
    jd_now = sim.current_jd()
    if jd_now is not None:
        y, m, d = calendar_date(jd_now)
        hh, mm, ss = clock_time(jd_now)
        imgui.text(f"date        {y:5d}-{m:02d}-{d:02d}")
        imgui.text(f"time     {hh:02d}:{mm:02d}:{ss:02d} UTC"
                   + ("  live" if sim.live() else ""))
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
        changed, val = imgui.slider_float("disk thickness", r.disk_h, 0.25, 3.0, "%.2f")
        if changed:
            r.disk_h = val
            sim.disk_h = val
        changed, val = imgui.checkbox("blackbody colour", r.blackbody)
        if changed:
            r.blackbody = val
        if imgui.is_item_hovered():
            imgui.set_tooltip("Colour the disk as a blackbody at its real temperature, "
                              "Doppler and gravitationally shifted.  Off: the old "
                              "artistic orange palette.  A disk this hot (~10^7 K) is "
                              "blue-white -- the visible colour of a blackbody stops "
                              "changing above ~10^5 K.")
        imgui.begin_disabled(not r.blackbody)
        changed, val = imgui.checkbox("peak shown at 6500 K", r.disk_tscaled)
        if changed:
            r.disk_tscaled = val
        if imgui.is_item_hovered():
            imgui.set_tooltip("Keep the disk's real temperature profile (T ~ r^-3/4) and "
                              "Doppler shifts, but show its hottest ring at 6500 K: white "
                              "inside, yellow, then red further out -- the colours a real "
                              "disk has when it is that hot, as around a supermassive "
                              "hole.  Off: the true ~10^7 K of this 10 M_sun hole, which "
                              "is blue-white everywhere.")
        imgui.end_disabled()
        changed, val = imgui.slider_float("spin  a*", sim.spin, -bhp.SPIN_MAX, bhp.SPIN_MAX,
                                          "%+.3f")
        if changed:
            sim.set_spin(val)
        if imgui.is_item_hovered():
            imgui.set_tooltip("J c / (G M^2).  + : the hole turns the same way as the "
                              "disk, - : against it.  0.998 is the most a disk can spin "
                              "a hole up to (Thorne 1974).  Moves the ISCO -- and so the "
                              "disk's inner edge -- from 4.5 r_s (retrograde) through "
                              "3 r_s (none) to 0.62 r_s (prograde).")
        kr = sim.kerr
        imgui.separator_text("geometry")
        imgui.text(f"mass         {BH_SOLAR_MASSES:.0f} M_sun")
        imgui.text(f"horizon      {kr.rh:4.2f} r_s  ({kr.rh * BH_RS_KM:6.1f} km)")
        imgui.text(f"photon orbit {kr.ph_co:4.2f} / {kr.ph_counter:4.2f} r_s (co / counter)")
        imgui.text(f"ISCO         {kr.isco:4.2f} r_s  ({kr.isco * BH_RS_KM:6.1f} km)")
        imgui.text(f"efficiency   {100.0 * kr.eta:4.1f} % of rest mass radiated")
        imgui.separator_text("disk (from the debris)")
        if r.disk_bright > 0.01:
            imgui.text(f"inner edge   {scene.disk_in:5.2f} r_s  (ISCO)")
            imgui.text(f"outer edge   {scene.disk_out:5.1f} r_s  (where the gas is)")
            imgui.text(f"accretion    {sim.mdot / bhp.M_SUN_KG:8.2e} M_sun/s"
                       f" = {sim.mdot_edd:7.1e} x Eddington")
            imgui.text(f"H/R          {sim.hr_visc:5.3f}  (from that rate)")
            imgui.text(f"T_max        {sim.t_max:8.2e} K")
            r_o = max(scene.disk_out, kr.isco)
            t_visc = 1.0 / max(sim.alpha * sim.hr_visc ** 2 * kr.omega(r_o), 1e-30)
            t_s = t_visc * BH_RS_KM / 299792.458
            imgui.text(f"drains in    ~{t_visc:7.0f} time units ({t_s * 1e3:.0f} ms)")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Viscous time at the outer edge, 1/(alpha (H/R)^2 Omega), "
                                  "with the H/R the accretion rate gives the disk.")
        else:
            imgui.text("no disk")
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
            # r_t = r_h (M_bh / M_star)^(1/3), with r_h the ball's half-mass
            # radius -- taken from the scale radius make_star actually builds
            # it with, so the two cannot drift apart again
            r_t = (STAR_HALF_MASS * cfg["r_star"]
                   * (0.5 / cfg["m_star"]) ** (1.0 / 3.0))
            beta = r_t / max(cfg["r_peri"], 1e-6)
            verdict = "full disruption" if beta >= 1.0 else "survives the pass"
            if cfg["r_peri"] < sim.kerr.rmb:
                verdict = "swallowed whole"
            col = imgui.ImVec4(0.5, 1.0, 0.6, 1.0) if beta >= 1.0 else imgui.ImVec4(1.0, 0.8, 0.4, 1.0)
            imgui.text(f"tidal radius {r_t:6.1f} r_s")
            imgui.text_colored(col, f"beta = r_t/r_p = {beta:4.2f}  {verdict}")
            # The star is drawn bigger than the hole, which is the right way round,
            # but it is nothing like the true ratio -- so say so rather than let
            # the picture imply otherwise.  See the preset's docstring.
            r_km = cfg["r_star"] * BH_RS_KM
            imgui.text(f"star radius {r_km:8.0f} km = {r_km / 696340.0:.4f} R_sun")
            t_eff = main_sequence_teff(cfg["m_star"] * SIM_MASS_TO_SOLAR)
            kind = ("M" if t_eff < 3900 else "K" if t_eff < 5300 else "G" if t_eff < 6000
                    else "F" if t_eff < 7500 else "A" if t_eff < 10000 else "B")
            imgui.text(f"surface T   {t_eff:8.0f} K  ({kind}-type main sequence)")
            imgui.text_colored(imgui.ImVec4(0.65, 0.7, 0.8, 1.0),
                               "(compact for its mass, so the disruption")
            imgui.text_colored(imgui.ImVec4(0.65, 0.7, 0.8, 1.0),
                               " happens where the lensing is visible)")

        imgui.separator_text("debris")
        changed, val = imgui.checkbox("self-gravity", sim.self_gravity)
        if changed:
            sim.self_gravity = val
        changed, val = imgui.checkbox("disk grows from debris", app.auto_disk)
        if changed:
            app.auto_disk = val
        changed, val = imgui.slider_float("circularisation", sim.circ, 0.01, 1.0, "%.3f",
                                          imgui.SliderFlags_.logarithmic)
        if changed:
            sim.circ = val
        changed, val = imgui.slider_float("viscosity alpha", sim.alpha, 0.01, 0.3, "%.3f",
                                          imgui.SliderFlags_.logarithmic)
        if changed:
            sim.alpha = val
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
        imgui.set_next_window_size(imgui.ImVec2(348, 520), imgui.Cond_.first_use_ever)
        imgui.begin("Solar System")
        changed, val = imgui.checkbox("body names  (N)", app.show_labels)
        if changed:
            app.show_labels = val
        changed, val = imgui.checkbox("orbit paths  (O)", app.show_orbits)
        if changed:
            app.show_orbits = val

        if scene.name == "Solar System":
            draw_date_controls(app)

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
                    if not app.sim.live():   # the wall clock owns the rate
                        app.sim.speed = min(app.sim.speed * 1.4, 16.0)
                elif ev.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS):
                    if not app.sim.live():
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

        if app.sim.paused:
            app.sim.hold_clock()
        else:
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
        renderer.bh_spin = app.sim.spin
        if renderer.disk_tscaled:
            renderer.log_tmax = math.log10(DISK_TPEAK_SHOWN)
        elif app.sim.t_max > 0.0:
            renderer.log_tmax = math.log10(app.sim.t_max)
        renderer.log_tstar = math.log10(app.sim.t_star)

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
