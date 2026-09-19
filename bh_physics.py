"""
Black hole physics used by the Gargantua scene in main.py.

Everything the black hole scene draws or integrates is derived here from the
hole's mass and spin and from the accretion rate the N-body debris actually
delivers -- nothing on the black hole side of main.py is a free "look" dial
any more.  Kept in its own module, with no GPU imports, so that every formula
can be checked on its own (see the self-test at the bottom: python
bh_physics.py).

Conventions
-----------
* Geometric units, G = c = 1.  Radii are given either in M (= GM/c^2, the
  gravitational radius) or in r_s (= 2M, the Schwarzschild radius, which is
  the world unit of the black hole scene); every function says which.
* Spin a = J / M^2, signed relative to the accretion flow: a > 0 means the
  hole turns the same way as the disk (prograde), a < 0 the opposite way
  (retrograde).  The disk always orbits the same way -- the way the star that
  feeds it was thrown in -- so flipping the sign of a flips the hole, not the
  disk.  |a| is capped at 0.998, Thorne's (1974) limit for a hole spun up by
  a radiating disk: the photons the disk's inner edge sends into the hole
  with negative angular momentum stop it ever reaching a = 1.

References
----------
Bardeen, Press & Teukolsky 1972, ApJ 178, 347       ISCO, photon orbit, r_mb
Bardeen 1970, Nature 226, 64                        spin-up by accretion
Page & Thorne 1974, ApJ 191, 499                    relativistic disk flux
Shakura & Sunyaev 1973, A&A 24, 337                 alpha disk, H/R, spherisation
Narayan & Yi 1995, ApJ 452, 710                     advective (thick) flow H/R
Artemova, Bjornsson & Novikov 1996, ApJ 461, 565    pseudo-Newtonian Kerr force
Blandford & Znajek 1977, MNRAS 179, 433             jet power from spin
Tchekhovskoy, Narayan & McKinney 2010/2011          BZ efficiency, MAD flux
Shimura & Takahara 1995, ApJ 445, 780               colour correction f_col
Chandrasekhar 1960, Radiative Transfer              limb darkening
Wyman, Sloan & Shirley 2013, JCGT 2(2)              CIE 1931 fit
"""

import math

import numpy as np

# --- SI constants (CODATA 2018) --------------------------------------------
G_SI = 6.67430e-11
C_SI = 2.99792458e8
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
SIGMA_SB = 5.670374419e-8
M_SUN_KG = 1.98892e30
# Electron-scattering opacity of solar-composition gas, 0.2 (1 + X) cm^2/g
# with hydrogen mass fraction X = 0.7, in m^2/kg.  The same opacity sets the
# Eddington luminosity and the Eddington-limited disk flux, so the two agree.
KAPPA_ES = 0.02 * (1.0 + 0.70)

SPIN_MAX = 0.998            # Thorne 1974


def clamp_spin(a):
    return max(-SPIN_MAX, min(SPIN_MAX, float(a)))


# ---------------------------------------------------------------------------
# Kerr geometry.  All radii in units of M unless the name says _rs.
# ---------------------------------------------------------------------------

def horizon(a):
    """Outer event horizon r_+ = M (1 + sqrt(1 - a^2))."""
    return 1.0 + math.sqrt(max(1.0 - a * a, 0.0))


def isco(a):
    """Innermost stable circular orbit (Bardeen, Press & Teukolsky 1972).
    Signed a: prograde (a > 0) shrinks it towards M, retrograde grows it
    towards 9 M."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0)
                                                + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    s = math.sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0))
    return 3.0 + z2 - math.copysign(s, a) if a != 0.0 else 6.0


def photon_orbit(a):
    """Circular equatorial photon orbit for photons co-rotating with the
    disk: 2M (1 + cos(2/3 arccos(-a)))."""
    return 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(max(-1.0, min(1.0, -a)))))


def marginally_bound(a):
    """Marginally bound circular orbit, r_mb = 2M - a + 2 sqrt(M (M - a)).
    A particle with E <= 1 that gets inside this radius has no turning point
    left: bound material that reaches it is captured, whatever it does next."""
    return 2.0 - a + 2.0 * math.sqrt(max(1.0 - a, 0.0))


def isco_energy(a):
    """Specific energy of the ISCO orbit, E = sqrt(1 - 2 M / (3 r_isco))."""
    return math.sqrt(1.0 - 2.0 / (3.0 * isco(a)))


def isco_angmom(a):
    """Specific angular momentum at the ISCO, in units of M (Bardeen 1970),
    positive in the sense the disk orbits."""
    return 2.0 / (3.0 * math.sqrt(3.0)) * (1.0 + 2.0 * math.sqrt(3.0 * isco(a) - 2.0))


def efficiency(a):
    """Radiative efficiency of a Novikov-Thorne disk, eta = 1 - E_isco:
    5.7% for a = 0, 32% at a = 0.998, 3.8% retrograde."""
    return 1.0 - isco_energy(a)


def omega_horizon(a):
    """Angular velocity of the horizon, Omega_H = a / (2 r_+), in 1/M."""
    return a / (2.0 * horizon(a))


def kepler_omega(r, a):
    """Angular velocity of a prograde circular equatorial orbit (BL time),
    Omega = 1 / (r^1.5 + a), r in M, result in 1/M."""
    return 1.0 / (np.asarray(r, dtype=np.float64) ** 1.5 + a)


def spin_up(m, a, dm0):
    """Accrete rest mass dm0 from the ISCO (Bardeen 1970).

    The hole gains the orbit's energy and angular momentum, not the rest
    mass: dM = E_isco dm0, dJ = l_isco M dm0.  What is missing from dM is
    exactly what the disk radiated.  Returns the new (M, a), a capped at the
    Thorne limit."""
    if dm0 <= 0.0:
        return m, a
    e = isco_energy(a)
    l = isco_angmom(a) * m
    j = a * m * m + l * dm0
    m_new = m + e * dm0
    return m_new, clamp_spin(j / (m_new * m_new))


# ---------------------------------------------------------------------------
# Pseudo-Newtonian force for the N-body integrator
# ---------------------------------------------------------------------------
#
#   F(r) = -GM / (r^(2 - beta) (r - r_H)^beta),   beta = r_isco / r_H - 1
#
# (Artemova, Bjornsson & Novikov 1996).  It puts the horizon and the ISCO at
# their true Kerr radii for any spin; for a = 0 it is exactly Paczynski-Wiita.
# Its potential integrates in closed form:
#
#   Phi(r) = -GM ((1 - r_H/r)^(1 - beta) - 1) / (r_H (beta - 1))
#
# (-> -GM/(r - r_s) for beta = 2, GM ln(1 - r_H/r)/r_H for beta = 1).

def abn_params(a, rs_world):
    """(r_H, beta) in world units for a hole whose r_s is rs_world."""
    rh = 0.5 * horizon(a) * rs_world
    beta = isco(a) / horizon(a) - 1.0
    return rh, beta


def abn_potential(r, gm, rh, beta):
    r = np.asarray(r, dtype=np.float64)
    x = np.clip(1.0 - rh / r, 1e-12, None)
    if abs(beta - 1.0) < 1e-6:
        return gm * np.log(x) / rh
    return -gm * (x ** (1.0 - beta) - 1.0) / (rh * (beta - 1.0))


def abn_vcirc(r, gm, rh, beta):
    r = np.asarray(r, dtype=np.float64)
    return np.sqrt(gm * r ** (beta - 1.0) / np.clip(r - rh, 1e-9, None) ** beta)


# ---------------------------------------------------------------------------
# Accretion disk
# ---------------------------------------------------------------------------

def eddington_luminosity(m_sun):
    """L_Edd = 4 pi G M c / kappa_es, in W."""
    return 4.0 * math.pi * G_SI * m_sun * M_SUN_KG * C_SI / KAPPA_ES


def page_thorne_shape(r, a):
    """Dimensionless relativistic flux profile of a Novikov-Thorne disk.

    F(r) = (3 Mdot c^2 / (8 pi r_g^2)) * page_thorne_shape(r, a), r in M.
    Zero at the ISCO (no torque there), -> (1 - sqrt(r_in/r)) / r^3 far
    out.  Page & Thorne 1974, eq. 15n, with F = Mdot f / (4 pi r)."""
    r = np.asarray(r, dtype=np.float64)
    x = np.sqrt(r)
    x0 = math.sqrt(isco(a))
    ac = math.acos(a)
    x1 = 2.0 * math.cos((ac - math.pi) / 3.0)
    x2 = 2.0 * math.cos((ac + math.pi) / 3.0)
    x3 = -2.0 * math.cos(ac / 3.0)
    out = np.zeros_like(x)
    ok = x > x0 * (1.0 + 1e-9)
    xs = x[ok]
    br = xs - x0 - 1.5 * a * np.log(xs / x0)
    for xi, xj, xk in ((x1, x2, x3), (x2, x1, x3), (x3, x1, x2)):
        br -= (3.0 * (xi - a) ** 2 / (xi * (xi - xj) * (xi - xk))
               * np.log((xs - xi) / (x0 - xi)))
    out[ok] = br / (xs ** 4 * (xs ** 3 - 3.0 * xs + 2.0 * a))
    return np.maximum(out, 0.0)


ADVECTIVE_HR = 0.53   # Narayan & Yi 1995 self-similar flow, gamma = 4/3, f = 1:
                      # c_s^2 = (2/7) v_K^2, so H/R -> sqrt(2/7) = 0.53
GAS_HR_FLOOR = 0.005  # gas-pressure supported thin disk, the floor as mdot -> 0


def scale_height(r, a, mdot_edd, eta):
    """Disk half-thickness over radius, r in M.

    Radiation-pressure support (Shakura & Sunyaev 1973):
        H = (3/2) (mdot / eta) M (1 - sqrt(r_in / r)),
    where mdot = Mdot / Mdot_Edd.  It is merged smoothly into the fully
    advective limit H/R = 0.53 that a super-Eddington flow saturates at,
    which is what keeps a disk fed at thousands of times Eddington a thick
    torus rather than a sphere many times bigger than its orbit."""
    r = np.asarray(r, dtype=np.float64)
    f = np.clip(1.0 - np.sqrt(isco(a) / np.maximum(r, 1e-9)), 0.0, None)
    h_rad = 1.5 * (mdot_edd / max(eta, 1e-6)) * f / np.maximum(r, 1e-9)
    hr = 1.0 / np.sqrt(1.0 / np.maximum(h_rad, 1e-30) ** 2 + 1.0 / ADVECTIVE_HR ** 2)
    return np.maximum(hr, GAS_HR_FLOOR)


def eddington_flux(r_m, m_kg, hr):
    """The most flux a disk surface at r can radiate before its own radiation
    pressure lifts it: F = c g_z / kappa with g_z = G M H / (r^2 + H^2)^1.5
    at the surface z = H.  Anything the disk is fed beyond that is advected
    into the hole or blown off, not radiated -- Shakura & Sunyaev's
    spherisation."""
    r_m = np.asarray(r_m, dtype=np.float64)
    gz = G_SI * m_kg * hr / (r_m * r_m * (1.0 + hr * hr) ** 1.5)
    return C_SI * gz / KAPPA_ES


def colour_correction(t_eff):
    """Electron scattering hardens a hot disk's spectrum: it looks like a
    blackbody at f_col T_eff, diluted by f_col^-4 (Shimura & Takahara 1995).
    f_col ~ 1.7 once scattering dominates the opacity (T above ~1e5 K), 1 in
    a cool disk; blended between in log T."""
    lt = np.log10(np.maximum(np.asarray(t_eff, dtype=np.float64), 1.0))
    w = np.clip((lt - 4.0) / 1.0, 0.0, 1.0)
    return 1.0 + 0.7 * w * w * (3.0 - 2.0 * w)


# ---------------------------------------------------------------------------
# Jets
# ---------------------------------------------------------------------------

BZ_KAPPA = 0.05       # field-geometry constant, 0.044-0.054 (Tchekhovskoy 2010)
PHI_MAD = 50.0        # horizon flux at which the disk is magnetically arrested
                      # (Gaussian units, Tchekhovskoy, Narayan & McKinney 2011)


def bz_efficiency(a, phi):
    """Blandford-Znajek jet power over Mdot c^2 (Tchekhovskoy et al. 2010):

        eta_jet = (kappa / 4 pi) phi^2 w^2 (1 + 1.38 w^2 - 9.2 w^4),
        w = Omega_H r_g / c = a / (2 r_+)

    phi is the dimensionless magnetic flux threading the horizon,
    Phi / sqrt(Mdot r_g^2 c).  No spin, no jet: the jet is the hole's own
    rotational energy being extracted through the field lines it drags."""
    w = omega_horizon(a)
    w2 = w * w
    return BZ_KAPPA / (4.0 * math.pi) * phi * phi * w2 * max(1.0 + 1.38 * w2 - 9.2 * w2 * w2, 0.0)


def jet_radius(z, rh):
    """Jet boundary: the paraboloidal field line (Blandford 1976) that threads
    the horizon at its equator, r (1 - cos theta) = r_H, i.e.
    R(z) = sqrt(r_H (2 z + r_H)).  Same units as z and rh."""
    return np.sqrt(rh * (2.0 * np.abs(z) + rh))


SYNC_ALPHA = 0.7      # optically thin synchrotron, F_nu ~ nu^-alpha


# ---------------------------------------------------------------------------
# Colour: blackbodies and power laws, through the CIE 1931 observer
# ---------------------------------------------------------------------------

def _lobe(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz_bar(lam_nm):
    """CIE 1931 2-degree matching functions, multi-lobe fit of Wyman, Sloan &
    Shirley (2013) -- within a percent or two of the tabulated curves."""
    l = np.asarray(lam_nm, dtype=np.float64)
    x = (1.056 * _lobe(l, 599.8, 37.9, 31.0) + 0.362 * _lobe(l, 442.0, 16.0, 26.7)
         - 0.065 * _lobe(l, 501.1, 20.4, 26.2))
    y = 0.821 * _lobe(l, 568.8, 46.9, 40.5) + 0.286 * _lobe(l, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(l, 437.0, 11.8, 36.0) + 0.681 * _lobe(l, 459.0, 26.0, 13.8)
    return x, y, z


XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])
_LUM = np.array([0.2126, 0.7152, 0.0722])
_LAM = np.linspace(380.0, 780.0, 161)          # nm
_DLAM = (_LAM[1] - _LAM[0]) * 1e-9             # m


def planck_lambda(lam_nm, t):
    lam = np.asarray(lam_nm, dtype=np.float64) * 1e-9
    xq = H_SI * C_SI / (lam * KB_SI * t)
    return 2.0 * H_SI * C_SI ** 2 / lam ** 5 / np.expm1(np.minimum(xq, 700.0))


def _spectrum_to_rgb(spec):
    """Spectral radiance (W m^-2 sr^-1 m^-1, sampled on _LAM) -> (linear sRGB
    chromaticity with unit luminance, luminance in cd/m^2)."""
    xb, yb, zb = cie_xyz_bar(_LAM)
    xyz = np.array([np.sum(spec * xb), np.sum(spec * yb), np.sum(spec * zb)]) * _DLAM * 683.0
    rgb = np.maximum(XYZ_TO_SRGB @ xyz, 0.0)
    lum = float(_LUM @ rgb)
    if lum <= 0.0:
        return np.array([1.0, 0.0, 0.0]), 1e-300
    return rgb / lum, xyz[1]


BB_LOGT_MIN, BB_LOGT_MAX, BB_N = 2.5, 9.5, 512


def blackbody_lut():
    """(BB_N, 4) float32 table over log10 T: linear-sRGB colour of a
    blackbody normalised to unit luminance, and log10 of its luminance in
    cd/m^2.  Past the top of the table a blackbody is deep in its
    Rayleigh-Jeans tail across the visible, where its colour stops changing
    and its luminance is simply proportional to T, so the shader extrapolates
    from the last entry."""
    out = np.zeros((BB_N, 4), dtype=np.float32)
    for i, lt in enumerate(np.linspace(BB_LOGT_MIN, BB_LOGT_MAX, BB_N)):
        rgb, y = _spectrum_to_rgb(planck_lambda(_LAM, 10.0 ** lt))
        out[i, :3] = rgb
        out[i, 3] = math.log10(max(y, 1e-300))
    return out


def blackbody_rgb(t_kelvin):
    """Unit-luminance linear sRGB of a blackbody, vectorised (for sprites)."""
    lut = _BB_CACHE if _BB_CACHE is not None else _bb_cache()
    lt = np.clip(np.log10(np.maximum(np.asarray(t_kelvin, dtype=np.float64), 1.0)),
                 BB_LOGT_MIN, BB_LOGT_MAX)
    f = (lt - BB_LOGT_MIN) / (BB_LOGT_MAX - BB_LOGT_MIN) * (BB_N - 1)
    i0 = np.clip(np.floor(f).astype(int), 0, BB_N - 2)
    w = (f - i0)[..., None]
    return lut[i0, :3] * (1.0 - w) + lut[i0 + 1, :3] * w


_BB_CACHE = None


def _bb_cache():
    global _BB_CACHE
    _BB_CACHE = blackbody_lut()
    return _BB_CACHE


def powerlaw_rgb(alpha):
    """Unit-luminance colour of an F_nu ~ nu^-alpha spectrum, i.e.
    F_lambda ~ lambda^(alpha - 2): the colour of optically thin synchrotron,
    which a Doppler shift scales in brightness but leaves the same colour."""
    return _spectrum_to_rgb((_LAM * 1e-9) ** (alpha - 2.0))[0]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name:44s} {got:12.6f}  (expect {want})")

    check("horizon a=0", horizon(0.0), 2.0, 1e-12)
    check("horizon a=0.998", horizon(0.998), 1.0632, 1e-3)
    check("ISCO a=0", isco(0.0), 6.0, 1e-9)
    check("ISCO a=0.998 (prograde)", isco(0.998), 1.2370, 1e-3)
    check("ISCO a=-1 (retrograde)", isco(-1.0), 9.0, 1e-9)
    check("ISCO a=0.5", isco(0.5), 4.2330, 1e-3)

    # Independent check of the ISCO: it is where the specific angular momentum
    # of the exact Kerr circular orbit is smallest.
    def kerr_l(r, a):
        return ((r * r - 2 * a * np.sqrt(r) + a * a)
                / (r ** 0.75 * np.sqrt(r ** 1.5 - 3 * np.sqrt(r) + 2 * a)))
    for a in (0.998, 0.7, -0.5, -0.998):
        rr = np.linspace(isco(a) * 0.9, isco(a) * 1.1, 40001)
        check(f"ISCO == min Kerr l(r), a={a:+.3f}", float(rr[np.argmin(kerr_l(rr, a))]),
              isco(a), 2e-4 * isco(a))
    check("photon orbit a=0", photon_orbit(0.0), 3.0, 1e-9)
    check("photon orbit a=1 (prograde)", photon_orbit(1.0), 1.0, 1e-9)
    check("photon orbit a=-1 (retrograde)", photon_orbit(-1.0), 4.0, 1e-9)
    check("r_mb a=0", marginally_bound(0.0), 4.0, 1e-12)
    check("r_mb a=-1", marginally_bound(-1.0), 3.0 + 2.0 * math.sqrt(2.0), 1e-9)
    check("eta a=0", efficiency(0.0), 1.0 - math.sqrt(8.0 / 9.0), 1e-9)
    check("eta a=0.998", efficiency(0.998), 0.3208, 2e-3)
    check("eta a=-0.998", efficiency(-0.998), 0.0380, 1e-3)
    check("l_isco a=0 = 2 sqrt 3", isco_angmom(0.0), 2.0 * math.sqrt(3.0), 1e-9)
    check("ABN beta a=0 -> Paczynski-Wiita", abn_params(0.0, 1.0)[1], 2.0, 1e-9)
    check("ABN potential a=0 == -GM/(r-rs)",
          float(abn_potential(10.0, 0.5, 1.0, 2.0)), -0.5 / 9.0, 1e-9)
    # ABN circular orbit is marginally stable exactly at the ISCO:
    # d(l^2)/dr = 0 there, l^2 = r^3 F(r)
    for a in (0.0, 0.9, -0.9):
        rh, be = abn_params(a, 1.0)
        rr = np.linspace(0.5 * isco(a) * 0.8, 0.5 * isco(a) * 1.2, 20001)
        l2 = rr ** 3 * 0.5 / (rr ** (2 - be) * (rr - rh) ** be)
        check(f"ABN angmom minimum at ISCO, a={a:+.1f} [r_s]",
              float(rr[np.argmin(l2)]), 0.5 * isco(a), 2e-4)
    # Page-Thorne closed form against its defining integral,
    #   F = Mdot/(4 pi r) * (-Omega')/(E - Omega L)^2 * int (E - Omega L) L' dr,
    # evaluated numerically with the exact Kerr circular-orbit E, L, Omega.
    for a in (0.0, 0.9, -0.7):
        def orb(r):
            d = r ** 0.75 * np.sqrt(r ** 1.5 - 3 * np.sqrt(r) + 2 * a)
            e = (r ** 1.5 - 2 * np.sqrt(r) + a) / d
            l = (r * r - 2 * a * np.sqrt(r) + a * a) / d
            return e, l, 1.0 / (r ** 1.5 + a)
        r = isco(a) + np.linspace(0.0, 40.0, 400001) ** 2 / 40.0
        e, l, om = orb(r)
        integ = (e - om * l) * np.gradient(l, r)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
        f = -np.gradient(om, r) / (e - om * l) ** 2 * cum / (4 * math.pi * r)
        for rt in (2.0 * isco(a), 30.0):
            k = int(np.searchsorted(r, rt))
            check(f"Page-Thorne vs integral, a={a:+.1f}, r={rt:5.2f}",
                  float(f[k] / (3 / (8 * math.pi) * page_thorne_shape(r[k], a))), 1.0, 2e-3)
    check("Page-Thorne zero at ISCO", float(page_thorne_shape(isco(0.3), 0.3)), 0.0, 1e-12)
    # Bardeen 1970 closed form, from a = 0:
    #   a = sqrt(2/3) (M0/M) (4 - sqrt(18 (M0/M)^2 - 2))
    m, a = 1.0, 0.0
    while m < 1.8:
        m, a = spin_up(m, a, 1e-5)
    y = 1.0 / m
    check("Bardeen spin-up at M = 1.8 M0", a,
          math.sqrt(2.0 / 3.0) * y * (4.0 - math.sqrt(18.0 * y * y - 2.0)), 2e-4)
    check("BZ efficiency a=0", bz_efficiency(0.0, 50.0), 0.0, 1e-12)
    print("BZ efficiency, MAD, a=0.99: %.2f  (Tchekhovskoy 2011: ~1.4)"
          % bz_efficiency(0.99, 47.0))
    lut = blackbody_lut()
    c5800 = lut[int((math.log10(5800) - BB_LOGT_MIN) / (BB_LOGT_MAX - BB_LOGT_MIN) * (BB_N - 1))]
    print("5800 K blackbody, linear sRGB:", np.round(c5800[:3], 3))
    print("1e7 K blackbody, linear sRGB: ", np.round(lut[-1, :3], 3),
          "(should be the blue-white Rayleigh-Jeans limit)")
    print("synchrotron alpha=0.7 colour: ", np.round(powerlaw_rgb(SYNC_ALPHA), 3))
    print("L_Edd(10 Msun) = %.3e W" % eddington_luminosity(10.0))
    print("ALL OK" if ok else "FAILURES")
    return ok


if __name__ == "__main__":
    _selftest()
