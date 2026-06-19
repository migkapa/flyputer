"""
visfield.py — LINK 3: the VISUAL-FIELD model.

Maps a world obstacle (bearing relative to the drone heading + angular size that grows as the
drone nears) onto the fly's REAL retinotopic L1 lamina (optic.optic()['l1g']), relays it
through the REAL L1->Mi1 wiring, and converts the resulting medulla activation into a looming
drive that fires the REAL Giant Fiber (DNp01) via swatter.run_loom.

Concrete reusable mapping (right eye, EYE='right'):
    azimuth  az in [-AZ_FOV, +AZ_FOV] deg  ->  u = 0.5 + 0.5*az/AZ_FOV      (clipped 0..1)
    elevation el in [-AZ_FOV, +AZ_FOV] deg ->  v = 0.5 + 0.5*el/AZ_FOV      (clipped 0..1)
    angular DIAMETER ang deg               ->  disc radius = 0.5*ang/(2*AZ_FOV) in (u,v)
    illuminated L1 columns                 =  { c : hypot(u_c-u, v_c-v) <= radius }

AZ_FOV=60 -> azimuth -60..+60 deg spans the full u axis 0..1 (validated).

HONEST: this is the ~789-column retinotopic L1 sheet (a brain's-eye hex sensor), not real
ommatidial optics. The vision -> looming step uses the magnitude of REAL L1->Mi1 medulla
activation as the looming-ramp strength driving the REAL LPLC2/LC4 detectors and REAL DNp01.
Everything in the path (L1 sheet, L1->Mi1 matrix, detectors, Giant Fiber) is real connectome;
only the scalar coupling medulla->ramp-peak is calibrated (toy LIF rates aren't biophysical).
"""
from __future__ import annotations

import numpy as np

import optic
import swatter

AZ_FOV = 60.0          # azimuth half-range (deg) mapped across the u axis
EL_FOV = 60.0          # elevation half-range (deg) mapped across the v axis
EYE_FILL_MEDULLA = 62000.0   # medulla activation when an obstacle fills the eye (calibration)
RAMP_PEAK_MAX = 40.0         # looming-ramp peak at full eye-fill


def bearing_to_uv(az_deg, el_deg=0.0):
    """Obstacle bearing (deg, relative to heading) -> retinotopic center (u,v) in [0,1]."""
    u = 0.5 + 0.5 * (az_deg / AZ_FOV)
    v = 0.5 + 0.5 * (el_deg / EL_FOV)
    return float(np.clip(u, 0, 1)), float(np.clip(v, 0, 1))


def angsize_to_radius(ang_deg):
    """Angular DIAMETER (deg) -> disc radius in (u,v). Full u span (1.0) == 2*AZ_FOV deg."""
    return 0.5 * (ang_deg / (2 * AZ_FOV))


def angsize_from_dist(real_radius, dist):
    """Angular DIAMETER (deg) of a physical sphere of `real_radius` at range `dist`."""
    return float(np.degrees(2 * np.arctan2(real_radius, dist)))


def illuminate(az_deg, ang_deg, el_deg=0.0, _O=None):
    """Return (center_u, center_v, radius, lit_column_indices) for an obstacle at this
    bearing/size painted on the L1 lamina."""
    O = _O or optic.optic()
    g = O["l1g"]
    cu, cv = bearing_to_uv(az_deg, el_deg)
    rad = angsize_to_radius(ang_deg)
    d = np.hypot(g[:, 0] - cu, g[:, 1] - cv)
    return cu, cv, rad, np.where(d <= rad)[0]


def medulla_activation(az_deg, ang_deg, el_deg=0.0, _O=None):
    """Paint the obstacle disc on REAL L1, relay through REAL L1->Mi1, return total medulla
    activation (the visual looming magnitude) + #lit L1 cols + #activated Mi1 cells."""
    O = _O or optic.optic()
    _, _, _, cols = illuminate(az_deg, ang_deg, el_deg, _O=O)
    drive = np.zeros(len(O["l1"]), dtype=np.float32)
    drive[cols] = 1.0
    mi = O["W"] @ drive
    return float(mi.sum()), int(len(cols)), int((mi > 0).sum())


def see_and_react(az_deg, real_radius, dist, el_deg=0.0, approach_ms=120, _O=None):
    """Full see->decide loop: obstacle of physical `real_radius` at bearing `az_deg`, range
    `dist`. Renders it on the eye, relays to medulla, drives the REAL looming detectors, and
    reports whether the REAL Giant Fiber (DNp01) fires. Returns a dict of every stage."""
    O = _O or optic.optic()
    ang = angsize_from_dist(real_radius, dist)
    medulla, n_lit, n_mi = medulla_activation(az_deg, ang, el_deg, _O=O)
    peak = RAMP_PEAK_MAX * (medulla / EYE_FILL_MEDULLA)
    r = swatter.run_loom(approach_ms=approach_ms, peak=max(peak, 0.1))
    return {
        "az_deg": az_deg, "dist": dist, "angsize_deg": ang,
        "n_l1_lit": n_lit, "n_mi1_active": n_mi, "medulla_act": medulla,
        "ramp_peak": peak, "gf_fires": r["gf_spike_t"] is not None,
        "gf_spike_t": r["gf_spike_t"],
    }


if __name__ == "__main__":
    O = optic.optic()
    print("LINK 3 visual field: %d L1 columns, AZ_FOV=%.0f deg -> u 0..1\n" % (len(O["l1"]), AZ_FOV))
    print("az->u:", {az: round(bearing_to_uv(az)[0], 3) for az in (-60, -30, 0, 30, 60)})
    print()
    print("%-7s %-9s %-7s %-8s %-9s %s" % ("dist", "angsize", "L1_lit", "medulla", "peak", "GF_fires"))
    for d in (12, 8, 5, 3, 2, 1.2):
        r = see_and_react(0.0, 1.7, d)
        print("%-7.1f %-9.1f %-7d %-8.0f %-9.1f %s"
              % (d, r["angsize_deg"], r["n_l1_lit"], r["medulla_act"], r["ramp_peak"], r["gf_fires"]))
